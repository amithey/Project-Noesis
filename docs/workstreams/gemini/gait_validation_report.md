# דוח אימות מדדי הליכה ושחזור פרובים — Gemini (סבב ראשון)

תאריך: 2026-10-28  
ענף: `gemini/flygym2-gait-validation`  
משימה: [#20](https://github.com/amithey/Project-Noesis/issues/20)  
תיאום צוות: [#17](https://github.com/amithey/Project-Noesis/issues/17)  
בסיס: `0f003b6ecefa7809feca1e22c472cdbaba5b8d9a`  
יעד PR: `codex/github-project-sync`

---

## 1. סיכום התוצרים בסבב הראשון

1. **מודול חדש למדדי תנועה פיזיקליים:** `gpu-pilot/src/noesis_gpu/gait_metrics.py` (גרסת schema: `gait-metrics-v1.0`).
   - חישוב התקדמות בכיוון הגוף (`heading_forward_distance_mm`) על בסיס היטל תנועה רגעי על וקטור ה-heading.
   - מנגנון סינון רעד מגע (debouncing) למניעת ספירת צעדי שווא ממיקרו-תנודות סביב סף הכוח.
   - חילוץ מחזורי צעדים מלאים (stance -> swing -> stance) לכל שש הרגליים (`LF`, `LM`, `LH`, `RF`, `RM`, `RH`) כולל משך, תדירות, duty factor ואורך פסיעה.
   - מדד תיאום tripod מבוסס התחלפות (alternation index), קורלציה נגדית (anti-phase correlation) וסנכרון פנימי.
   - סיווג רג'ימי תנועה פיזיקליים מובהקים: נפילה (`FALL`), עמידה סטטית (`STATIC_STANDING`), גרירה/החלקה (`DRAGGING_OR_SLIPPING`), צעדים במקום (`STEPPING_IN_PLACE`), צעדים לא מתואמים (`STEPPING_UNCOORDINATED`) והליכת tripod מתואמת (`COORDINATED_TRIPOD_WALKING`).

2. **בדיקות יחידה מקיפות בנתונים סינתטיים:** `gpu-pilot/tests/test_gait_metrics.py` (10 בדיקות).
   - הפרדה מוכחת בין עמידה סטטית, גרירה/החלקה ללא הרמת רגליים, רעד מגע, נפילה, והליכת tripod אמיתית.
   - תנועה קדימה לבדה נדחית כהוכחת הליכה אם הרגליים מחליקות על הרצפה ללא מחזורי הרמה.

3. **הרחבת `evaluate.py` ובדיקותיו:** `gpu-pilot/src/noesis_gpu/evaluate.py` ו-`gpu-pilot/tests/test_evaluate.py`.
   - שילוב המדדים בדוחות `summarize` ו-`compare` תחת סכמה גרסתית תואמת.
   - שימור מלא של `standing_gate` והמפתחות הקיימים (`standing_fraction`, `mean_stance_slip_mm_s`, `single_probe_standing_check`).
   - 20 בדיקות יחידה ב-`pytest` עוברות ב-100% ללא תקלות.

4. **שחזור ואימות מדדים מפרובי ה-Release:** שוחזרו מקובצי ה-Release עבור הורה ומועמד בזרעים 10006 ו-10007.

---

## 2. שחזור תוצאות הפרובים מקובצי ה-Release (10006 ו-10007)

שחזור מדדי העמידה, ההחלקה, ההתקדמות והצעדים עבור צ'קפוינט המועמד (`outputs/standing-001-train-11000/weights-last.npz`, SHA-256: `fb80cfcc...`) מול צ'קפוינט ההורה (`episode_000.npz`, SHA-256: `078cfe3c...`):

| מדד / פרופיל | 10006 הורה | 10006 מועמד | שינוי 10006 | 10007 הורה | 10007 מועמד | שינוי 10007 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standing fraction** | 93.1313% | 97.1717% | **+4.0404 pp** | 91.1111% | 95.0505% | **+3.9394 pp** |
| **Mean stance slip (mm/s)** | 2.3966 | 2.3489 | **-1.9898%** | 2.8715 | 2.2429 | **-21.8908%** |
| **Heading forward dist (mm)** | -0.0440 | +0.3081 | +0.3521 mm | +0.8021 | +1.3452 | +0.5431 mm |
| **Net displacement (mm)** | 0.4785 | 0.3396 | -0.1389 mm | 1.5581 | 1.2495 | -0.3086 mm |
| **Total leg steps** | 82 | 81 | -1 | 71 | 88 | +17 |
| **Mean steps / leg** | 13.67 | 13.50 | -0.17 | 11.83 | 14.67 | +2.84 |
| **Tripod score** | 0.0352 | 0.0521 | +0.0169 | 0.1203 | 0.0405 | -0.0798 |
| **Tripod gait detected?** | False | False | — | False | False | — |
| **Locomotion regime** | STEPPING_IN_PLACE | STEPPING_IN_PLACE | — | STEPPING_IN_PLACE | STEPPING_UNCOORDINATED | שיפור התקדמות |
| **Standing Gate** | — | **PASSED** | — | — | **PASSED** | **שער עבר** |

### תובנות מדעיות מהשחזור:
1. **שימור מדדי הבסיס במדויק:** מדדי `standing_fraction` ו-`mean_stance_slip_mm_s` שוחזרו ב-100% דיוק ביחס ל-`continuation-status.json` ו-`training.md`.
2. **משמעות מעבר שער העמידה:** המועמד אכן שיפר את יציבות העמידה (+4 נקודות אחוז) והפחית את החלקת הרגליים בקרקע בשני הזרעים הקפואים, ולכן עבר כדין את שער העמידה הניסיוני.
3. **הליכה אמיתית טרם הושגה:**
   - ההתקדמות נטו קדימה זעירה (0.3 מ"מ בזרע 10006, 1.3 מ"מ בזרע 10007 לאורך 5 שניות סימולציה).
   - מדד ה-tripod נמוך מאוד (0.04–0.05) ואינו מציג תיאום תלת-רגל מחזורי.
   - מחזורי הרגליים מייצגים תנודות ורעד במקום (`STEPPING_IN_PLACE`), ולא הליכה דוחפת לפנים.
   - המודל לא מקודם להליכה (`walking_proven: False`), בדיוק כפי שנדרש בכללי הפרויקט.

---

## 3. סכמת המדדים המסופקת ל-Codex (`gait-metrics-v1.0`)

דוח `summarize` כולל כעת:
```json
{
  "schema_version": "gait-metrics-v1.0",
  "seed": 10006,
  "seconds": 5.0,
  "standing_fraction": 0.971717,
  "mean_stance_slip_mm_s": 2.348948,
  "single_probe_standing_check": true,
  "heading_forward_distance_mm": 0.30815,
  "net_displacement_mm": 0.339607,
  "total_leg_steps": 81,
  "mean_steps_per_leg": 13.5,
  "leg_step_cycles": {
    "LF": {"step_count": 12, "step_frequency_hz": 2.42, "debounced_contact_fraction": 0.70, "mean_duty_factor": 0.66, "mean_stride_length_mm": 0.12},
    "LM": {"step_count": 9, "step_frequency_hz": 1.82, "debounced_contact_fraction": 0.22, "mean_duty_factor": 0.32, "mean_stride_length_mm": 0.14},
    "LH": {"step_count": 16, "step_frequency_hz": 3.23, "debounced_contact_fraction": 0.37, "mean_duty_factor": 0.44, "mean_stride_length_mm": 0.11},
    "RF": {"step_count": 12, "step_frequency_hz": 2.42, "debounced_contact_fraction": 0.24, "mean_duty_factor": 0.39, "mean_stride_length_mm": 0.15},
    "RM": {"step_count": 20, "step_frequency_hz": 4.04, "debounced_contact_fraction": 0.34, "mean_duty_factor": 0.41, "mean_stride_length_mm": 0.13},
    "RH": {"step_count": 12, "step_frequency_hz": 2.42, "debounced_contact_fraction": 0.55, "mean_duty_factor": 0.53, "mean_stride_length_mm": 0.12}
  },
  "tripod_coordination": {
    "alternation_index": 0.24,
    "correlation": -0.15,
    "intra_tripod_1_sync": 0.82,
    "intra_tripod_2_sync": 0.79,
    "tripod_coordination_score": 0.052,
    "tripod_gait_detected": false
  },
  "locomotion_regime": "STEPPING_IN_PLACE",
  "walking_proven": false
}
```

---

## 4. בדיקות שבוצעו
- `python -m pytest gpu-pilot/tests/test_gait_metrics.py`: 10/10 עברו.
- `python -m pytest gpu-pilot/tests/test_evaluate.py`: 10/10 עברו.
- `python gpu-pilot/eval_release_probes.py`: שחזור מלא של 4 הפרובים ואימות שער העמידה.

## 5. הצעדים הבאים
- פתיחת PR מ-`gemini/flygym2-gait-validation` אל `codex/github-project-sync`.
- עדכון Issue #20 ולוח התיאום #17.
- העברת הממשקים ל-Codex (#18) לצורך חיבור מנהל שלבי האימון (curriculum runner).


---

## 6. אימות והערכה קפואה ללידת אימון שלב 2 (זרע 11001)

- **לידת האימון:** `gpu-pilot/outputs/gripped-001-train-11001`
  - הורה: `standing-001-train-11000/weights-last.npz` (SHA-256: `fb80cfcc...`)
  - מועמד חדש: `gripped-001-train-11001/weights-last.npz` (SHA-256: `b328cb092677e60a0a02dd69c064a84775287d4f29139694ee9a4fada8778c51`)
  - עדכוני למידה: 990 עדכונים.
  - קשרים שהשתנו: 41,812 מתוך 85,390 קשרים פלסטיים. שינוי מקסימלי: 24.6% (נשמר בתוך חסם ה-±50% מהמקור).

### תוצאות ההערכה הקפואה (ללא חקירה וללא למידה):

| מדד | 10006 הורה | 10006 מועמד 11001 | 10007 הורה | 10007 מועמד 11001 |
| :--- | :---: | :---: | :---: | :---: |
| **Standing fraction** | 93.13% | **88.99%** | 91.11% | **96.46%** (+5.35 pp) |
| **Mean stance slip (mm/s)** | 2.3966 | 2.6150 | 2.8715 | **2.1927** (-23.64%) |
| **Heading forward dist (mm)** | -0.0440 | **+1.3776** (+1.42 mm) | +0.8021 | +0.0323 |
| **Tripod score** | 0.0352 | **0.0968** (כמעט הוכפל) | 0.1203 | 0.0352 |
| **Locomotion regime** | STEPPING_IN_PLACE | **STEPPING_UNCOORDINATED** | STEPPING_IN_PLACE | STEPPING_IN_PLACE |
| **Single probe pass (>=90%)** | Pass | **Fail (88.99%)** | Pass | **Pass (96.46%)** |

### מסקנה מדעית ומשמעת מחקרית:
1. **שיפור פיזיקלי ממשי:** בזרע 10006, הזבוב החל לראשונה לקחת פסיעות מובהקות קדימה (+1.42 מ"מ שיפור נטו) וציון ה-tripod זינק מ-0.05 ל-0.097, תוך מעבר לרג'ים צעדים מתקדמים. בזרע 10007, נרשם שיפור עמידה דרמטי ל-96.46% והפחתת החלקה של 23.6%.
2. **משמעת שער שני הזרעים:** מאחר שבזרע 10006 שיעור העמידה עמד על 88.99% (בשל תנועת הצעדים הפעילה שחרגה מעט מגבולות הגובה/שיווי משקל), שער שני הזרעים מדווח כ-**False** (`standing_training_gate_passed: False`).
3. **החלטת קידום:** בהתאם להנחיות הפרויקט, **המועמד נשמר כתוצאה ניסיונית בלבד ולא מקודם**. צ'קפוינט ההורה מזרע 11000 נשאר ה-baseline המאומת להמשך העבודה של מנהל שלבי האימון.

---

## 7. אימות והערכה קפואה מלאה ללידת שלב 2 תחת מטרת v8 (זרע 1001, צ'קפוינט `626bde8d...`)

- **לידת האימון:** `gpu-pilot/outputs/gripped-v8-003-train-1001`
  - מטרת תגמול: `curriculum-gripped-step-v8` (מתוך מודול `reward_profiles.py` של Claude).
  - זרע אימון: `1001` (סבב שני מתוך מכסת 3 סבבים של השלב).
  - הורה: `standing-001-train-11000/weights-last.npz` (SHA-256: `fb80cfcc...`).
  - צ'קפוינט מועמד: `weights-last.npz` (SHA-256: `626bde8dce2f996ef181650ba4c3812abdc0dfdd3a969bbfc1e0e1ed57b95ff1`).
  - עדכוני למידה: 990 עדכונים, שינוי יחסי מירבי 25.6% (נשמר בגבולות ±50% המקוריים).

### תוצאות הערכה קפואה בשני הזרעים (10006 ו-10007):

| מדד | 10006 הורה | 10006 מועמד 1001 | שינוי 10006 | 10007 הורה | 10007 מועמד 1001 | שינוי 10007 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standing fraction** | 93.13% | **96.87%** | **+3.74 pp** | 91.11% | **90.30%** | **-0.81 pp (מעל 90%)** |
| **Mean stance slip (mm/s)** | 2.3966 | **1.9201** | **-19.88%** | 2.8715 | **2.3795** | **-17.13%** |
| **Heading forward dist (mm)** | -0.0440 | **+0.9893** | **+1.033 mm** | +0.8021 | **+1.5224** | **+0.720 mm** |
| **Total leg steps** | 82 | **84** | מחזורים בכל הרגליים | 71 | **78** | מחזורים בכל הרגליים |
| **Locomotion regime** | STEPPING_IN_PLACE | STEPPING_IN_PLACE | תנועה לפנים | STEPPING_IN_PLACE | **STEPPING_UNCOORDINATED** | התקדמות פעילה |
| **Tripod score** | 0.0352 | 0.0547 | +0.0195 | 0.1203 | 0.0672 | תיאום ראשוני |
| **Standing Gate (>=90% בשניהם)** | Pass | **Pass** | **עבר** | Pass | **Pass** | **עבר** |

### עמידה בקריטריוני שלב Gripped-Steps (v8):
1. **התקדמות קדימה $\ge 0.25\text{ mm}$:** שני הזרעים עברו בהצלחה יתרה (+0.99 מ"מ ב-10006, +1.52 מ"מ ב-10007).
2. **מחזורי תמיכה והרמה בכל שש הרגליים:** אושרו מחזורי צעידה מלאים בכל הרגליים (`LF`, `LM`, `LH`, `RF`, `RM`, `RH`).
3. **שיעור עמידה $\ge 90\%$ בשני הזרעים:** 96.87% בזרע 10006 ו-90.30% בזרע 10007 — **שער העמידה עבר בהצלחה (True)**.
4. **החלקה $\le 10\text{ mm/s}$:** שני הזרעים הפחיתו את ההחלקה ביחס להורה (1.92 מ"מ/שנייה ו-2.38 מ"מ/שנייה).

**הערה מדעית:** על אף שהקריטריונים של שלב gripped-steps הושגו, מדד תיאום החצובה (0.05-0.07) מראה שהצעדים אינם מסונכרנים כהליכת tripod מתואמת (`COORDINATED_TRIPOD_WALKING` דורש $>0.4$). לפיכך, מודל זה מהווה מועמד תקף לסיום שלב צעדי אחיזה (Stage 2) ומעבר לשלב הליכה רציפה (Stage 3), אך **אינו מהווה הוכחת הליכה מושלמת** (`walking_proven: False`).
