# מסמך מסירת עבודת החושים ל-Claude/Opus — 1 בנובמבר 2026

**מוסר**: Gemini (#20)  
**מקבל**: Claude / Opus (#19)  
**בסיס קוד מחויב**: `4a78e0b` בענף `gemini/flygym2-gait-validation` (PR #22)  
**תיקיית עבודה**: `C:/dev/noesis-gemini-gait`  

---

## 1. רכיבי הקוד והמפרט שנמסרים

1. **מתאם דחף נוירונלי ([`external_senses.py`](file:///C:/dev/noesis-gemini-gait/gpu-pilot/src/noesis_gpu/external_senses.py))**:
   - מתאם Poisson drive גנרי מבוסס ערוצים (`ChannelSpec`, `ExternalSensoryAdapter`).
   - עומד ב-100% מחוזה `merge_drive`:
     - דחיית `NaN`, `Inf`, מספרים מרוכבים ובוליאנים.
     - אי-מוטציה של מערכי הקלט.
     - שילוב דטרמיניסטי ללא כפילויות (overlap rejection).
     - הבטחת אפס עירור ספונטני כאשר הערוץ כבוי או מתחת לסף (`enabled=False` או `signals=None`).
   - 21 בדיקות יחידה בנתונים סינתטיים עוברות ב-[`test_external_senses.py`](file:///C:/dev/noesis-gemini-gait/gpu-pilot/tests/test_external_senses.py).

2. **דוח מיפוי מקיף ([`sensory_mapping_and_validation_report.md`](file:///C:/dev/noesis-gemini-gait/docs/workstreams/gemini/sensory_mapping_and_validation_report.md))**:
   - מיפוי 17,082 אפקטורים חושיים ב-FlyWire (3,921 פרופריוספטורים ב-`SensoryMap`, 1,053 נוירוני שיווי משקל ב-`BalanceMap`, 8,156 במתאמי קוד שאינם מחוברים לסימולציה).

3. **הצעת ניסוי כיול מיושר ([`sensory_rectification_experiment_proposal.md`](file:///C:/dev/noesis-gemini-gait/docs/workstreams/gemini/sensory_rectification_experiment_proposal.md))**:
   - פרוטוקול להשוואת Baseline (76Hz טוני) מול Candidate (מיושר) בהורה `9fd7d411`.

---

## 2. מקור יחידות הכוח ב-FlyGym 2.1 (Force Units Provenance)
* תאוצת הכובד: $g = -9810.0\text{ mm/s}^2$.
* מסת גוף כוללת: $M = 0.0010243$.
* משקל כולל של הזבוב בסימולציה: $W = M \cdot g \approx 10.05$ **יחידות כוח סימולציה גולמיות ($F_{\text{sim}}$)**.
* עומס מבחן של 5.0 הוא $5.0\ F_{\text{sim}}$ (~50% ממשקל הגוף הכולל), ולא יחידות SI ישירות (mN).

---

## 3. דגשים מתמטיים ופיזיקליים קריטיים לשילוב החושי בלמידה (L1/L2)

1. **מסגור דחף ה-76Hz והבחנת שיפוע (Gain vs Offset)**:
   - דחף ה-76Hz במנוחה נובע ממרכוז הסיגמואיד באפס היסט: $\text{sig}(0) = 0.5$.
   - **שים לב לשינוי ה-Gain במעבר ל-tanh**:
     - שיפוע הסיגמואיד באפס: $\frac{d}{dx} \sigma(x / 0.3) |_{x=0} = \frac{1}{4 \cdot 0.3} \approx 0.833$.
     - שיפוע ה-tanh באפס: $\frac{d}{dx} \tanh(x / 0.3) |_{x=0} = \frac{1}{0.3} \approx 3.333$ (שיפוע תלול פי 4!).
     - כאשר מתכננים בדיקה מבודדת של הסרת דחף המנוחה, יש לקחת בחשבון את התלילות או לכייל את ה-gain בהתאם כדי לבודד offset מ-slope.

2. **חלוקה אנטומית לשיוך נוירון-לרגל**:
   - Prothoracic (T1) -> רגליים קדמיות (LF / RF).
   - Mesothoracic (T2) -> רגליים אמצעיות (LM / RM).
   - Metathoracic (T3) -> רגליים אחוריות (LH / RH).
   - עדיפות לשם העצב (`Nerve`), ורק בהיעדרו שימוש בצד הסומה (`Soma side`).

3. **בסיס תגובה פיזיקלי קבוע**:
   - כפי שאושר מול Codex, יש להימנע מ-PCA עצמאי בכל לידה בעת ניתוח קורלציית הלמידה, עקב היפוכי סימן וסיבוב אקראיים. מומלץ להשתמש בבסיס אנטומי קבוע או במטריצת היטל קפואה מההורה.

---

## 4. גבולות אחריות והמשך העבודה
* **Gemini**: ממשיך להחזיק בבעלות על מדדי ההליכה הפיזיקליים (`gait_metrics.py`), סכמת ההערכה (`evaluate.py`), ובביקורת פיזיקלית בלתי-תלויה של צ'קפוינטים ופרובים (כפי שבוצע ב-S1).
* **Claude / Opus**: מוביל את שילוב אותות החישה בתוך לולאת הלמידה והתגמולים בסביבתו המבודדת, ללא שינוי בקובצי הריצה המשותפים של Codex.
