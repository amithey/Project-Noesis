# H1: thorax height/contact/coordinate audit

Offline worker5.625s;22existing cases,9998stored training rewards replay exactly. Source/input/aggregate checks pass, exit0/empty, lease released. No model/dynamics/neural steps or threshold changes.

Position is global thorax body-frame-origin z in mm; floor plane z=0. Contact norms are not whole-body COM/vertical support forces. Upper1.5mm gate is present in both physical mask and v7training reward. Fixed-input scorer boundary jumps1.5333..1.6 over2e-6mm; this is not a simulated trajectory. Existing pass/fail verdicts remain unchanged.

|Bank|Train seed|Eval seed|Standing %|High-only %|Excess median mm|>=3claws %|Median abs vz mm/s|Longest high-only s|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|parent|None|10006|96.2626|3.1313|0.1359|70.9677|4.3056|0.1100|
|parent|None|10007|94.4444|3.9394|0.0261|2.5641|1.0096|0.0950|
|parent|None|10010|89.2929|5.9596|0.0457|86.4407|2.2321|0.1450|
|parent|None|10011|91.9192|6.0606|0.0193|48.3333|0.9590|0.1550|
|child|1004|10006|40.9091|4.0404|0.0527|10.0000|1.1588|0.1850|
|child|1004|10007|94.1414|3.3333|0.0384|57.5758|2.5360|0.0800|
|child|1004|10010|96.7677|2.1212|0.0139|23.8095|0.8601|0.0450|
|child|1004|10011|93.9394|5.1515|0.0128|58.8235|0.9476|0.1100|
|parent-own|1004|1004|89.1919|10.2020|0.0251|51.4851|1.0885|0.1500|
|child-own|1004|1004|98.8889|0.5051|0.0058|20.0000|0.7146|0.0250|
|child|1005|10006|97.6768|1.4141|0.0542|100.0000|2.1851|0.0700|
|child|1005|10007|87.6768|11.1111|0.1202|67.2727|1.9215|0.3500|
|child|1005|10010|94.2424|2.5253|0.1166|84.0000|1.0362|0.1250|
|child|1005|10011|90.8081|7.4747|0.0657|41.8919|1.9867|0.1050|
|parent-own|1005|1005|96.6667|2.2222|0.2141|72.7273|4.0410|0.1100|
|child-own|1005|1005|94.9495|4.4444|0.0621|34.0909|1.9525|0.0850|
|child|1006|10006|91.8182|7.1717|0.0263|67.6056|1.1734|0.1100|
|child|1006|10007|94.6465|4.4444|0.0641|84.0909|1.2956|0.1000|
|child|1006|10010|83.8384|12.3232|0.0352|55.7377|1.2537|0.1950|
|child|1006|10011|95.5556|2.2222|0.0378|68.1818|0.7863|0.1100|
|parent-own|1006|1006|94.0404|5.6566|0.0496|33.9286|1.9143|0.1400|
|child-own|1006|1006|86.4646|10.4040|0.0515|64.0777|1.8555|0.1900|

Limits:

- 5ms samples cannot exclude flight/contact loss between samples.
- Recorded42active joint angles omit passive tarsus states; no exact physical pose replay claimed.
- No whole-body COM, contact normals/points or complete solver state stored; no certified balance/overextension conclusion.
- Boundary probes change only scorer inputs, not physics or the reward used in training.
- No rescoring to success, new gate, promotion or claim that height explains the sustained1004failure.
