"""Bounded file-editing Claude/Gemini workers; independent tests, no automatic merge."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import team_workflow as review

OWNED = {
    'claude': ['gpu-pilot/src/noesis_gpu/retention_audit.py',
               'gpu-pilot/tests/test_retention_audit.py',
               'docs/workstreams/claude/retention-contract.md'],
    'antigravity': ['gpu-pilot/src/noesis_gpu/external_senses.py',
                    'gpu-pilot/tests/test_external_senses.py',
                    'docs/workstreams/gemini/external-senses-contract.md'],
}


def git(work, *args):
    return subprocess.check_output(['git', '-C', str(work), *args], encoding='utf-8').strip()


def changed_paths(work):
    tracked = git(work, 'diff', '--name-only', 'HEAD').splitlines()
    untracked = git(work, 'ls-files', '--others', '--exclude-standard').splitlines()
    return set(tracked + untracked) - {'.agents/agents/noesis-developer/agent.md', 'WORK_ORDER.md'}


def validate_changes(work, provider):
    changed = changed_paths(work)
    extra = changed - set(OWNED[provider])
    if extra:
        raise ValueError(f'Changes outside assignment: {sorted(extra)}')
    if changed != set(OWNED[provider]):
        raise ValueError(f'Missing deliverables: {sorted(set(OWNED[provider]) - changed)}')
    for name in OWNED[provider]:
        path = work / name
        if path.is_symlink() or not path.is_file() or not path.stat().st_size:
            raise ValueError(f'Invalid deliverable: {name}')
    return {name: hashlib.sha256((work / name).read_bytes()).hexdigest() for name in sorted(changed)}


def commands(provider, work):
    if provider == 'claude':
        return review.cli(provider) + ['--safe-mode', '-p',
            'Implement the assigned files described on stdin using file tools.',
            '--tools', 'Read,Write,Edit,Glob,Grep', '--allowedTools', 'Read,Write,Edit,Glob,Grep',
            '--permission-mode', 'acceptEdits', '--no-session-persistence', '--output-format', 'json']
    # Session-only policy. No global setting or blanket permission bypass.
    settings = {'permissions': {'deny': ['command(*)', 'unsandboxed(*)', 'read_url(*)',
                    'execute_url(*)', 'mcp(*)', 'write_file(.git/)', 'read_file(.git/)']}}
    return review.cli(provider) + ['--agent', 'noesis-developer', '--mode', 'accept-edits',
        '--model', 'gemini-3.8-flash-medium', '--settings', json.dumps(settings),
        '--print-timeout', '480s', '--input-format', 'stream-json', '--output-format', 'stream-json']


def update(state, task_id, **values):
    allowed = {'status','result','error','started','finished'}
    if not values or not values.keys() <= allowed:
        raise ValueError('Invalid task update')
    with review.database(state) as db:
        db.execute('UPDATE tasks SET ' + ','.join(k+'=?' for k in values) + ' WHERE id=?',
                   [*values.values(), task_id])


def render(state):
    with review.database(state) as db:
        rows = [dict(r) for r in db.execute('SELECT id,status,attempts,result,error FROM tasks ORDER BY id')]
    review.dump(state/'status.json', rows)
    lines = ['# Parallel development status', '', 'Provider file edits → ownership check → independent pytest. No automatic merge.', '']
    for r in rows:
        lines.extend([f"## {r['id']}: {r['status']}", '', r['result'] or r['error'] or 'Pending', ''])
    (state/'report.md').write_text('\n'.join(lines), encoding='utf-8')


def execute(state, task, config):
    provider = task['provider']
    work = Path(config['workspaces'][provider])
    folder = state/'attempts'/task['id']/str(task['attempts']+1)
    folder.mkdir(parents=True, exist_ok=False)
    update(state, task['id'], status='developing', started=time.time())
    with review.database(state) as db:
        db.execute('UPDATE tasks SET attempts=attempts+1 WHERE id=?', (task['id'],))
    try:
        if git(work, 'rev-parse', 'HEAD') != config['base_commit']:
            raise ValueError('Worker base changed; explicit coordinator review required')
        if changed_paths(work):
            raise ValueError('Worker workspace already contains changes; do not overwrite')
        prompt = (work/'WORK_ORDER.md').read_text(encoding='utf-8')
        (folder/'prompt.txt').write_text(prompt, encoding='utf-8')
        with (folder/'stdout.json').open('w', encoding='utf-8') as out, (folder/'stderr.log').open('w', encoding='utf-8') as err:
            proc = subprocess.Popen(commands(provider, work), cwd=work, stdin=subprocess.PIPE,
                stdout=out, stderr=err, text=True, encoding='utf-8', shell=False)
            review.dump(folder/'process.json', {'pid':proc.pid, 'started':time.time()})
            try:
                payload = json.dumps({'event':'user','message':{'content':prompt}}, ensure_ascii=False)+'\n' if provider == 'antigravity' else prompt
                proc.communicate(payload, timeout=510)
            except BaseException:
                if os.name == 'nt':
                    subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'], capture_output=True)
                else:
                    proc.kill()
                proc.wait()
                raise
        raw = (folder/'stdout.json').read_text(encoding='utf-8')
        diagnostic = raw + (folder/'stderr.log').read_text(encoding='utf-8')
        if review.quota_problem(diagnostic):
            raise BlockingIOError(review.quota_problem(diagnostic))
        if review.authentication_problem(diagnostic):
            raise PermissionError(review.authentication_problem(diagnostic))
        if proc.returncode:
            raise RuntimeError(f'Provider exit {proc.returncode}; inspect local attempt')
        response = review.parse_response(provider, raw)
        hashes = validate_changes(work, provider)
        review.dump(folder/'deliverable-hashes.json', hashes)
        update(state, task['id'], status='testing', result=response)
        env = os.environ.copy()
        root = Path(config['root'])
        env.update(PYTHONUTF8='1', NOESIS_SOURCE=config['core_source'],
            PYTHONPATH=os.pathsep.join([str(work/'gpu-pilot/src'), config['core_source'],
                str(root/'gpu-pilot/arrow-runtime'), str(root/'gpu-pilot/runtime')]))
        test = OWNED[provider][1]
        test_command = [config['python'], '-m', 'pytest', test, '-q',
            '--basetemp', str(folder/'pytest-temp'), '-o', 'cache_dir='+str(folder/'pytest-cache')]
        with (folder/'tests.log').open('w', encoding='utf-8') as log:
            checked = subprocess.run(test_command, cwd=work, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, timeout=120)
        if checked.returncode:
            raise RuntimeError('Independent synthetic tests failed; inspect tests.log')
        # Hash again: tests must not rewrite deliverables.
        if validate_changes(work, provider) != hashes:
            raise ValueError('Deliverables changed while testing')
        update(state, task['id'], status='ready_for_review', result=response+'\n\nIndependent tests:\n'+
               (folder/'tests.log').read_text(encoding='utf-8'), finished=time.time())
    except Exception as exc:
        status = 'blocked_quota' if isinstance(exc,BlockingIOError) else 'blocked_auth' if isinstance(exc,PermissionError) else 'failed'
        update(state, task['id'], status=status, error=f'{type(exc).__name__}: {exc}', finished=time.time())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--state', type=Path, required=True)
    ap.add_argument('action', choices=['run','status'])
    args = ap.parse_args()
    state = args.state.resolve()
    if args.action == 'status':
        render(state)
        print((state/'status.json').read_text(encoding='utf-8'))
        return
    with review.coordinator_lock(state):
        config = json.loads((state/'config.json').read_text(encoding='utf-8'))
        with review.database(state) as db:
            db.execute("UPDATE tasks SET status='interrupted',error='Explicit inspection required before retry' WHERE status IN ('developing','testing')")
            tasks = [dict(r) for r in db.execute("SELECT * FROM tasks WHERE status='queued'")]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(execute,state,t,config) for t in tasks]
            while not all(f.done() for f in futures):
                render(state)
                time.sleep(2)
            for f in futures:
                f.result()
        render(state)


if __name__ == '__main__':
    main()
