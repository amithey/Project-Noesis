# Claude diagnostic assignment — 1 November 2026

הנחיית משתמש חדשה,1בנובמבר2026: "תן לקלוד את העבודה". Claude הוא כעת מוביל סבב האבחון שהוצע; Codex נשאר סוקר ומשלב. זו העברת בעלות על האבחון, לא בקשה לעוד וריאציית אימון או מיזוג משקולות.

התחל עכשיו בעבודה קלה, בענף ו־checkout מבודדים משלך, בלי לבקש אישור להחלטות שגרתיות. קרא את TEAM_CHAT הראשי דרךhelper ואת מסמכי התיאום העדכניים בענףcodex/training-pilot-20261028, החל מ־320e9d8. אשר ב־#19 וביומן מצב/branch/checkout/SHA/קבצים לפני מימוש. אין לערוך ענפי Codex/Gemini או קובצי runtime משותפים; הכן מודול אבחון חדש ו־patch מוצע במידת הצורך.

תוצר ראשון: PR קטן עם פרוטוקול, אבחון וקוד בדיקות שמפרידים את השרשרת חקירה→ירי עצבי→הפעלת שרירים→פקודות→גוף במשקולות קבועות וללא למידה. השתמש בהורה9fd7d41177bf2c745809401c3b142a832661d1a18d0b117b30cb02aea63fea3c ובפינים הניסיוניים הקיימים(core42c2076), או פרט מראש מדוע הצעת פרוטוקול דורשת מקור אחר. תכנן שלוש זרועות מזווגות: חקירה כבויה, אותו מסלול עם זרם אפס, וחקירה פעילה. אמת קודם שהרישום עצמו ואפס־ההתערבות אינם משנים את הבסיס, את זרמי האקראיות או את המשקולות. רשום פעילות/פקודות בפועל, לא רק תנועת רגליים; בלי מחולל הליכה, סדר רגליים או פקודות גוף חיצוניות. קבע אופק מלא, זרעים, מדדים, תקציב ותנאי עצירה לפני כל הקצאה. זרעים10,11,200,201,202 נשמרים.

תוצר שני ונפרד: בקרה חיובית קטנה למנגנון הלמידה עם קשר מושתל וידוע בין פעילות לתגמול, ובקרה שלילית תואמת, מבחן לאחר כיבוי למידה וקריטריון שנקבע מראש. שמור אתMotorPlasticityהאמיתי ומגבלותיו. זהו מבחן מנגנון סינתטי; אל תציג אותו כהצלחה פיזיקלית או כמבחן יחיד שמזהה חד־משמעית את סיבת הכשל.

סגור תיקוניjoint-probe/ties,sidecarוניסוחnull הנוגעים להכרעות שלך. קבל את מסירתGemini2c3dca9 אך בדוק חתימות לפני קבלה. S1 הואv7מולv9במשך25שניות זהה;control25אומן25שניות והוערך5. cases.jsonהואסיכום,לאrawNPZ. אין עדיין הוכחה שחישה או רעש טוני גרמו לכשל. הצעת מקור חישה/hook יכולה להיות תוצר המשך נפרד; אל תשלבfoot_load,rectificationותגמול לרגל בניסוי אחד.

הקצאה נוכחית: קוד, פרוטוקול, ניתוח קל ובדיקות זעירות בלבד. לבדיקה מקומית קבע מראש תקציב של עד600שניות מצטברות, בליfull-connectome/FlyGymאו סימולציה כבדה; עצור בכשל טכני/חריגת תקציב ודווח. מסורCodexפרוטוקול תקציב להרצה הכבדה; אין הקצאה כבדה בהודעה זו. ההרצה הפיזיקלית תשתמש ב־leaseהאטומי המשותף הקנוני ובבדיקת משאבים חדשה. אין דריסת הורה/checkpoint,שינוי ספים/פינים/learning-ruleאו קידום מודל. עדכן תוצאה/חסם ב־TEAM וב־#19,קוד וקומפקטים ב־GitHub וארטיפקטים עםSHA256.

Authoritative assignment: https://github.com/amithey/Project-Noesis/issues/19#issuecomment-6555793675
Delivery: submitted directly to Claude's existing Project-Noesis chat by Codex under the user's explicit instruction. Agent acknowledgment must be observed separately; task submission alone is not proof of implementation.

Observed acknowledgment: TEAM `a46f95658578`. Claude reports own checkout `C:/dev/noesis-claude-diag`, branch `claude/chain-diagnostic`, start `15e2bfe`; accepts the three deliverables and cumulative600s tiny-test budget. Existing chat displays Working. No completed diagnostic or physical result is claimed.
