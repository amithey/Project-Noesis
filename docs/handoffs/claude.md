# מסירה ל־Opus / Claude — 30 באוקטובר 2026

המשתמש ביקש לסדר את העבודה. אתה אחראי ללמידה ולשימור; Codex אחראי לעמידה/הליכה, תהליך האימון, ההרצות והאינטגרציה. משימתך: https://github.com/amithey/Project-Noesis/issues/19 .

קרא AGENTS, docs/project-status-20261030.md, docs/team-training-plan.md ו־TEAM_CHAT הראשי באמצעות `python scripts/team_chat.py read`. אשר קבלה ביומן ובמשימה, עם checkout/branch/SHA/קבצים. המקור שנצפה לאחרונה: claude/learning-retention ב־bbbfd7765172c6cafc30023d1786d8cb3fcdf7ac; אין בכך הוכחה שעובד רץ. אל תערוך ענף או קבצים של עמית.

התוצר הבא הוא PR קטן לתיקון כלי credit-timing בלבד:

1. אות learning_signal המתוקן עבר אצל Codex התאמה מדויקת לכל990 השורות הפעילות בשלושת הלוגים הקיימים. שמור זאת ואת הוצאת warmup.
2. בטל תלות בסימן/סיבוב של PCA שנלמד בנפרד לכל לידה: השתמש בבסיס משותף מתועד או בסטטיסטיקה חסינה לטרנספורמציה, ובדוק רגרסיית sign/rotation. דוגמת הנגד ב־docs/credit-timing-review-counterexample.json.
3. תקן שיוך רגליים מפורש לפי סוג תא/מיפוי מוטורי מוכר; אל תסיק רגל CNS מכל prefix.221/389 נוירונים מוטוריים בלוגים הושמטו בגלל תוויות .NO_CONS. פרסם כיסוי, unknowns ורגרסיות; ראה docs/leg-assignment-audit-20261030.json.
4. תאר circular-shift reference כתיאורי כל עוד stationarity/exchangeability לא הוכחו; אין p-value מכויל או טענה סיבתית מהקשר הסגור.
5. בדוק קודם את3 הלוגים הקיימים (D3seed1000 ו־C2seed1001/1002), עם hashes, התאמת שורות והקשר. אל תשנה LR, eligibility, reward או משקולות במסגרת התיקון.

מסור commit, פקודות ותוצאות בדיקה, תוצאות שלושת הלוגים ומגבלות ב־#19 וב־TEAM_CHAT. אין הקצאה לשמונה לידות חדשות או לסימולציית full-connectome. בדיקות יחידה/ניתוח קל מותרות; אין אישור לפעילות של עובד רק משום שנשלחה הודעה. Codex סוקר לפני שילוב; המועמדים נשארים נפרדים.
