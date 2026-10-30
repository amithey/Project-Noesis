# מסירה ל־Gemini — 30 באוקטובר 2026

המשתמש ביקש לסדר את העבודה. אתה אחראי לחושים/תגובה לסביבה ולאימות מדידה עצמאי; Codex אחראי לעמידה/הליכה ולהרצות, ו־Opus/Claude ללמידה. משימתך: https://github.com/amithey/Project-Noesis/issues/20 .

קרא AGENTS, docs/project-status-20261030.md, docs/team-training-plan.md ו־TEAM_CHAT הראשי באמצעות `python scripts/team_chat.py read`. אשר קבלה עם branch/checkout/SHA/קבצים. ענף המדדים האחרון שנבדק היה gemini/flygym2-gait-validation ב־99db49f. דיווחת על feature/habitat-in-view, אך docs/workstreams/gemini/sensory_mapping_and_validation_report.md לא נמצא שם. ה־checkout הראשי מלוכלך; אל תכתוב בו או תשנה שינויים של אחרים. הצע ענף מועמד ו־checkout מבודדים לפני מימוש נוסף; שם מוצע אינו ענף פעיל.

התוצר הבא הוא חבילת ראיות חישה קטנה ומחויבת ב־Git:

1. פרסם SHA ונתיב דוח נגישים, קוד baseline/candidate מדויק, parent/metadata/source hashes, seed ואורך הרצה. צרף ארטיפקטים מזווגים ו־SHA256 manifest ב־Release.
2. שמור את תיקון יחידות העומס: raw simulation force units, לא mN ללא המרה מפורשת. תעד את מקור mass/gravity והבדל בין גירוי סינתטי לעומס פיזיקלי שנמדד.
3. הצג76Hz במצב מנוחה כדחף טוני של המודל; ירידה82%בספייקים אינה הוכחה לעמידה/הליכה טובה יותר. בדיקת מוח200 מילישניות seed42 אינה בדיקת שימור פיזיקלית.
4. בדוק no-signal/disabled baseline, תקינות ערכים ומיפוי אנטומי; תעד unknowns ותוויות .NO_CONS בתיאום עם Opus. שמור את מדדי standing/slip/heading/cycles ואת השערים הקיימים.
5. הצע ניסוי פיזיקלי מזווג המשנה רק כיול חישה, מהורה משותף, בנפרד משינוי למידה. הצעה אינה הקצאת הרצה או אישור מיזוג.

מסור commit, בדיקות בפועל, hashes ומגבלות ב־#20 וב־TEAM_CHAT. אין סימולציה כבדה מוקצית, כולל בדיקת מוח full-connectome; בדיקות יחידה/עבודה קלה מותרות.9fd7d411 הוא הורה ניסיוני, לא validated-best. אין קידום שלב, מיזוג משקולות או שינוי startup רגיל; Codex סוקר לפני שילוב.
