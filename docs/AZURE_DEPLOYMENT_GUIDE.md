# دليل رفع مشروع REAL_i (REAL_i) على سحابة مايكروسوفت آژور (Microsoft Azure Deployment Guide)

بصفتي خبيرًا عالميًا في بنية وحلول مايكروسوفت آژور (Azure Solutions Architect)، قمت بإعداد هذا الدليل الشامل والمفصل خطوة بخطوة لمساعدتك في رفع تطبيق **REAL_i** (الخاص بمشروع التخرج) بالكامل على السحابة ليعمل بكفاءة عالية وبأقل تكلفة ممكنة، مع ضمان استقرار نماذج الذكاء الاصطناعي ومعالجة النصوص.

---

## 1. نظرة عامة على البنية البرمجية السحابية (Cloud Architecture)

يتكون مشروع REAL_i من المكونات التالية، وهنا نوضح كيف سيتم استضافتها على Azure:

```mermaid
graph TD
    User([المستخدم / الطالب / المعلم]) -->|يتصفح| Frontend[واجهة المستخدم - React Vite]
    Frontend -->|Azure Static Web Apps| CDN[شرايين التوزيع العالمية CDN]
    Frontend -->|طلبات API| Backend[FastAPI Backend - Docker Container]
    Backend -->|Azure App Service for Containers| BackendService[بيئة تشغيل FastAPI]
    BackendService -->|تخزين دائم للملفات وقواعد البيانات المحلية| AzureFiles[(Azure File Share)]
    BackendService -->|بيانات المستخدمين والمقاييس| MongoDB[(MongoDB Atlas Cloud)]
    BackendService -->|استدعاء النماذج اللغوية| OpenRouter[OpenRouter API / gpt-4o-mini]
```

### المكونات السحابية المستخدمة:
1. **Azure Static Web Apps (SWA):** لاستضافة واجهة React + Vite مجانًا مع دعم شهادة SSL مجانية وتوزيع محتوى عالمي سريع (CDN).
2. **Azure App Service (Web App for Containers):** لاستضافة الخلفية (FastAPI Backend) داخل حاوية Docker.
   > [!IMPORTANT]
   > لا ينصح باستخدام بيئة تشغيل Python العادية (Native Python runtime) على App Service لأن المشروع يحتوي على مكتبات ثقيلة جدًا مثل PyTorch و PaddleOCR و EasyOCR و Docling والتي يستغرق تثبيتها وقتًا طويلًا وتفشل عادة بسبب نفاد الذاكرة أثناء التثبيت. استخدام Docker هو الحل الذهبي.
3. **Azure Container Registry (ACR):** لتخزين صور Docker (Docker Images) الخاصة بالخلفية بشكل آمن.
4. **Azure Storage Account (Azure Files):** لربط مساحة تخزين سحابية مشتركة داخل حاوية الـ Backend.
   > [!CAUTION]
   > حاويات Docker هي حاويات مؤقتة (Stateless). إذا تم إعادة تشغيل الخادم، فستفقد كل الملفات المرفوعة وقواعد بيانات الفيكتور المحلية (ChromaDB / Qdrant). ربط Azure Files بمسار الحاوية يضمن حفظ ملفات PDFs وقاعدة بيانات الفيكتور بشكل دائم (Persistent Storage).
5. **MongoDB Atlas (Azure Region):** قاعدة بيانات الكائنات المستضافة سحابيًا (لديك بالفعل خادم جاهز ومحدد في ملف `.env`).

---

## 2. الخطوة 1: تهيئة الحاوية (Containerization) محليًا

لقد قمنا بإنشاء ملف [Dockerfile](file:///d:/REAL_i/Dockerfile) وملف [.dockerignore](file:///d:/REAL_i/.dockerignore) في جذر المشروع لتجهيز الخلفية. 

### مميزات الـ Dockerfile المرفق:
* يستخدم نسخة مخففة من بايثون (`python:3.10-slim`).
* يثبت متطلبات النظام اللازمة لتشغيل التعرف الضوئي على الحروف (OCR) ومعالجة الصور ومكتبات PDF.
* يثبت نسخة الـ CPU من PyTorch لتوفير المساحة واستهلاك الذاكرة.
* **تحميل مسبق للنماذج (Critical Optimization):** يقوم بتحميل نموذج التضمين `gte-multilingual-base` ونموذج إعادة الترتيب `bge-reranker-v2-m3` أثناء بناء الصورة (Build Phase). هذا يمنع آژور من محاولة تحميل جيجابايتات من البيانات عند إقلاع الحاوية لأول مرة مما يؤدي لفشل التشغيل بسبب انتهاء الوقت المحدد (Timeout).

### لتجربة بناء الصورة وتشغيلها محليًا للتأكد:
1. افتح Terminal في جذر المشروع `d:\REAL_i`.
2. قم ببناء الصورة:
   ```bash
   docker build -t raaed-backend .
   ```
3. قم بتشغيل الصورة للتجربة (مع تمرير ملف `.env`):
   ```bash
   docker run -p 5000:5000 --env-file .env raaed-backend
   ```
4. تأكد من أن الـ API يعمل بطلب `http://localhost:5000/`.

---

## 3. الخطوة 2: إنشاء حساب التخزين السحابي (Azure Storage Account)

الحاويات على Azure لا تحفظ البيانات عند إعادة التشغيل. سنقوم بإنشاء مساحة تخزين سحابية لربطها بـ Docker.

1. سجل الدخول إلى [بوابة آژور (Azure Portal)](https://portal.azure.com/).
2. اضغط على **Create a resource** ثم ابحث عن **Storage account** واضغط **Create**.
3. قم بملء البيانات:
   * **Subscription:** اختر اشتراكك (مثال: Azure for Students).
   * **Resource Group:** أنشئ مجموعة جديدة باسم `raaed-group`.
   * **Storage account name:** اختر اسمًا فريدًا صغيرًا (مثال: `raaedstorage`).
   * **Region:** اختر نفس منطقتك (يفضل غرب أوروبا `West Europe` أو شرق الولايات المتحدة `East US`).
   * **Performance:** اختر **Standard**.
   * **Redundancy:** اختر **Locally-redundant storage (LRS)** لتقليل التكلفة القصوى.
4. اضغط على **Review + create** ثم **Create**.

### إنشاء مجلدات التخزين المشتركة (File Shares):
بعد اكتمال الإنشاء، اذهب إلى الحساب السحابي المنشأ (`raaedstorage`):
1. من القائمة الجانبية، اضغط على **File shares** (تحت قسم *Data storage*).
2. اضغط على **+ File share** وأنشئ وحدة تخزين باسم: `raaed-files` (لتخزين ملفات الـ PDFs المرفوعة للمساقات).
3. اضغط على **+ File share** مرة أخرى وأنشئ وحدة تخزين باسم: `raaed-db` (لتخزين قاعدة بيانات الفيكتور Qdrant/Chroma).
4. حدد السعة المقترحة بـ **10 GB** أو أكثر (يمكن زيادتها لاحقًا بسهولة وتكلفتها بسيطة جدًا بالـ Cent).

---

## 4. الخطوة 3: إنشاء سجل الحاويات (Azure Container Registry - ACR)

سنقوم بإنشاء سجل خاص لرفع صورة الـ Docker Backend عليه لتستطيع خدمة App Service سحبها.

1. في بوابة Azure، ابحث عن **Container registries** واضغط **Create**.
2. قم بتعبئة البيانات:
   * **Resource Group:** اختر `raaed-group`.
   * **Registry name:** اختر اسمًا فريدًا (مثال: `raaedregistry`).
   * **Location:** نفس موقع الـ Storage account.
   * **SKU:** اختر **Basic** (كافية تمامًا لمشاريع التخرج ورخيصة جدًا).
3. اضغط **Review + create** ثم **Create**.
4. بعد الإنشاء، اذهب إلى السجل ومن القائمة الجانبية اضغط على **Access keys** وقم بتمثيل الخيار **Admin user** ليصبح **Enabled** (سيعطيك هذا اسم مستخدم وكلمة مرور ستحتاجها لتسجيل الدخول ورفع الصورة).

### رفع صورة Docker إلى السجل (ACR):
افتح الـ Terminal في جهازك محليًا واكتب الأوامر التالية (استبدل القيم ببيانات السجل الخاص بك):

```bash
# 1. تسجيل الدخول إلى سجل Azure (سيطلب منك كلمة المرور من صفحة Access Keys)
docker login raaedregistry.azurecr.io

# 2. وضع وسم (Tag) للصورة المحلية لتتوافق مع آژور
docker tag raaed-backend raaedregistry.azurecr.io/raaed-backend:v1

# 3. رفع الصورة إلى السحاب
docker push raaedregistry.azurecr.io/raaed-backend:v1
```

---

## 5. الخطوة 4: إنشاء خدمة استضافة الخلفية (Azure App Service for Containers)

الآن سنقوم بحجز بيئة تشغيل سحابية لتشغيل حاوية الخلفية (FastAPI).

1. في بوابة Azure، ابحث عن **App Services** واضغط **Create** -> **Web App**.
2. قم بتعبئة البيانات في تبويب **Basics**:
   * **Resource Group:** اختر `raaed-group`.
   * **Name:** اختر اسمًا لـ API الخلفية (مثال: `raaed-api`). سيكون الرابط العام: `https://raaed-api.azurewebsites.net`.
   * **Publish:** اختر **Container**.
   * **Operating System:** اختر **Linux**.
   * **Region:** نفس المنطقة السابقة.
   * **Pricing Plan (Critical):** اضغط على *Change size* لتحديد خطة التشغيل.
     > [!WARNING]
     > **تنبيه هام جدًا لخبير السحاب:** لا تختار خطة Free (F1) أو Shared (D1) لأنها توفر ذاكرة عشوائية (RAM) حجمها 1GB فقط. نماذج التضمين والمطابقة والـ OCR في المشروع تستهلك حوالي 2 إلى 3 جيجابايت RAM عند تحميلها.
     > **الخيار الموصى به لمشروع تخرج:**
     > اختر خطة **Basic B2** أو **B3** (تأتي بذاكرة 4GB أو 8GB RAM مع معالج مميز) أو خطة **Premium V3 (P1v3)** إذا كان لديك رصيد طلابي كافٍ (Student Credits).
3. اضغط على تبويب **Container** في الأعلى:
   * **Image Source:** اختر **Azure Container Registry**.
   * **Registry:** اختر `raaedregistry`.
   * **Image:** اختر `raaed-backend`.
   * **Tag:** اختر `v1`.
4. اضغط **Review + create** ثم **Create**.

### ربط مساحات التخزين المستمرة (Path Mappings):
لكي يتم ربط مجلدات التخزين السحابي التي أنشأناها في الخطوة 2 بداخل الحاوية:
1. اذهب إلى صفحة الـ Web App المنشأ (`raaed-api`).
2. من القائمة الجانبية، اضغط على **Configuration** ثم اختر التبويب الثاني في الأعلى وهو **Path mappings**.
3. اضغط على **+ New Azure Storage Mount**:
   * **Name:** `files-mount`
   * **Configuration Options:** Basic
   * **Storage Accounts:** اختر `raaedstorage`
   * **Storage Type:** Azure Files
   * **Storage Container (File Share):** اختر `raaed-files`
   * **Mount Path:** اكتب المسار داخل الحاوية بالضبط: `/app/src/assets/files`
4. اضغط على **+ New Azure Storage Mount** مرة أخرى:
   * **Name:** `db-mount`
   * **Configuration Options:** Basic
   * **Storage Accounts:** اختر `raaedstorage`
   * **Storage Type:** Azure Files
   * **Storage Container (File Share):** اختر `raaed-db`
   * **Mount Path:** اكتب المسار داخل الحاوية بالضبط: `/app/src/assets/database`
5. اضغط على **Save** في أعلى الصفحة لحفظ الإعدادات وإعادة تشغيل التطبيق.

### تكوين متغيرات البيئة (Environment Variables):
من نفس صفحة **Configuration** ولكن في التبويب الأول **Application settings**:
اضغط على **+ New application setting** لإدخال كل المتغيرات الموجودة بملف الـ `.env` الخاص بك:

| اسم المتغير (Name) | القيمة المقترحة (Value) | ملاحظات |
| :--- | :--- | :--- |
| `WEBSITES_PORT` | `5000` | يخبر آژور أن الحاوية تستمع على منفذ 5000 ويقوم بتوجيه طلبات الويب الخارجية إليه تلقائيًا. |
| `WEBSITES_CONTAINER_START_TIME_LIMIT` | `1800` | مهم جدًا! يرفع وقت مهلة إقلاع الحاوية إلى 30 دقيقة، ليتيح للخادم تحميل النماذج الكبيرة من Hugging Face وفحص المجلدات في المرة الأولى دون أن يفشل الإقلاع. |
| `MONGODB_URL` | `mongodb+srv://goharhany9_db_user:z15599We2k3SULur@cluster0...` | رابط قاعدة بيانات أطلس المستضافة سحابيًا. |
| `MONGODB_DATABASE` | `reali-db` | اسم قاعدة البيانات. |
| `OPENAI_API_KEY` | `sk-or-v1-e97ed021412bb92163485...` | مفتاح OpenRouter أو OpenAI الخاص بك. |
| `OPENAI_API_URL` | `https://openrouter.ai/api/v1` | رابط واجهة OpenRouter البرمجية. |
| `GENERATION_MODEL_ID` | `openai/gpt-4o-mini` | نوع النموذج المستخدم للمحادثات والاختبارات. |
| `VECTOR_DB_BACKEND` | `QDRANT` | نوع قاعدة الفيكتور. |
| `VECTOR_DB_PATH` | `reali_qdrant_db` | اسم المجلد الذي سينشأ بداخل `/app/src/assets/database/`. |
| `VECTOR_DB_DISTANCE_METHOD` | `cosine` | طريقة حساب المسافة للفيكتور. |
| `GOOGLE_SPREADSHEET_ID` | `1wMtkgZgVr2vIZvcxBRIePRbKF...` | معرف ورقة Google لعميل الأدمين. |
| `GOOGLE_CLIENT_ID` | `725493678609-93vhjrdv2pt59p...` | حساب OAuth Google Client ID. |
| `GOOGLE_CLIENT_SECRET` | `GOCSPX-0wrra70CQ2wiQ5-joh...` | حساب OAuth Google Client Secret. |
| `ASSISTANT_WEBHOOK_URL` | `https://raaed-api.azurewebsites.net/api/v1/agent/webhook/task` | **مهم جدًا:** استبدل localhost برابط السيرفر الجديد على Azure لكي تتمكن webhook الخاصة بالـ Task من إرسال التنبيهات للخلفية مباشرة سحابيًا. |

*اضغط على **Save** في الأعلى ثم تأكيد الإجراء لتطبيق المتغيرات وإعادة التشغيل.*

---

## 6. الخطوة 5: رفع واجهة المستخدم (Vite React Frontend)

سنقوم باستخدام **Azure Static Web Apps (SWA)** لرفع الواجهة مباشرة عبر GitHub، وهي الطريقة الأسرع والمجانية بالكامل.

### أ. دفع الكود إلى مستودع GitHub
تأكد من أن الكود بالكامل مرفوع على مستودع GitHub الخاص بك: `Gohar-Hany/REAL_i-Graduation-Project`.

### ب. تهيئة رابط الـ API في الواجهة
عند بناء الواجهة في آژور، ستبحث عن رابط الخلفية. لقد قمنا بتعديل ملف [api.js](file:///d:/REAL_i/frontend/src/services/api.js) ليقبل متغير البيئة `VITE_API_URL`.

### ج. إنشاء Static Web App في Azure:
1. في بوابة Azure، ابحث عن **Static Web Apps** واضغط **Create**.
2. تعبئة البيانات:
   * **Resource Group:** اختر `raaed-group`.
   * **Name:** `raaed-frontend`
   * **Plan type:** اختر **Free** (متاح مجانًا بنسبة 100%).
   * **Region:** اختر منطقة قريبة (مثال: `West Europe` أو `East US 2`).
3. اضغط على **Sign in with GitHub** وقم بتسجيل الدخول وتفويض آژور.
4. اختر بيانات المستودع الخاص بك:
   * **Organization:** حسابك.
   * **Repository:** `REAL_i-Graduation-Project`.
   * **Branch:** فرعك الرئيسي (مثال: `main` أو `master`).
5. في قسم **Build Details** (هام جدًا لتطبيق React Vite):
   * **Build Presets:** اختر **Vite**.
   * **App location:** اكتب `/frontend` (لأن مجلد الواجهة موجود كـ Sub-folder داخل المشروع).
   * **Api location:** اتركه فارغًا.
   * **Output location:** اكتب `dist`.
6. اضغط **Review + create** ثم **Create**.

### د. تكوين رابط الخلفية لنسخة الإنتاج:
عند إنشاء الـ Static Web App، سيقوم تلقائيًا بإنشاء ملف إعداد لـ GitHub Actions في مستودعك لبناء ورفع الموقع. لتمرير رابط الـ API أثناء البناء:
1. اذهب إلى صفحة الـ **Static Web App** المنشأ في بوابة Azure.
2. من القائمة الجانبية اضغط على **Configuration**.
3. في تبويب **Application settings** اضغط على **+ Add**:
   * **Name:** `VITE_API_URL`
   * **Value:** `https://raaed-api.azurewebsites.net/api/v1` (رابط الـ App Service API الذي قمنا بإنشائه بالخطوة 4).
4. اضغط **Save**.
5. سيقوم محرك GitHub Actions بإعادة بناء الموقع وحفظ الرابط بداخل ملفات الـ Javascript في نسخة الإنتاج بشكل تلقائي بالكامل!

---

## 7. الخطوة 6: معالجة الـ CORS في الخلفية

حاليًا، يسمح الـ Backend باستقبال الطلبات فقط من روابط الـ Localhost مثل `http://localhost:5173`. عندما يرتفع الموقع على رابط آژور، سيرفض المتصفح الطلبات بسبب حماية الـ CORS.

لتعديل ذلك بشكل دائم ومرن على السيرفر:
1. انسخ رابط الواجهة السحابي الخاص بك من الـ Static Web App (سيكون شيئًا مثل `https://proud-sea-12345.azurestaticapps.net`).
2. يمكنك تعديل ملف `src/main.py` لإضافة هذا الرابط إلى قائمة الروابط المسموح بها في `CORSMiddleware`.
   أو يمكنك تعديله ليسمح بجميع الروابط في بيئة الإنتاج كالتالي:
   ```python
   # في ملف src/main.py حوالي السطر 108
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"], # يسمح لجميع الواجهات بالاتصال، وهو الأسهل لمشاريع التخرج
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

---

## 8. الخطوة 7: التحقق والتشغيل والاختبار

بمجرد اكتمال الرفع السحابي للخلفية والواجهة:
1. **فحص الخلفية:** افتح المتصفح على الرابط: `https://raaed-api.azurewebsites.net/api/v1/admin/health` وتأكد من استجابة النظام بـ `{"status": "ok"}` أو ما شابه ومطابقة اتصال قاعدة البيانات بنجاح.
2. **فحص الـ Logs:** إذا حدثت أي مشكلة أثناء تشغيل الخلفية، يمكنك متابعة الـ Logs مباشرة عبر:
   * اذهب إلى الـ App Service السحابي -> **Log Stream** لمشاهدة المخرجات الفورية أثناء التشغيل.
3. **فحص الواجهة:** افتح رابط الـ Static Web App وجرب إجراء العمليات الأساسية:
   * رفع ملف مساق جديد والتأكد من استقراره في Azure Files.
   * طلب إنشاء كويز للتأكد من قيام نماذج الذكاء الاصطناعي بمعالجة النصوص واسترجاع الفيكتور بشكل ممتاز.

---

## 9. نصائح خبير آژور لتقليل النفقات (Cost Optimization)

كخالب تخرج، قد تنفد أرصدة الطلاب بسرعة إذا لم تكن حذرًا. اتبع هذه النصائح الاحترافية لتوفير المال:
1. **إيقاف الخدمة عند عدم الاستخدام (Stop App Service):**
   عند انتهاء فترات العمل على المشروع أو أثناء النوم، اذهب إلى صفحة الـ Web App (`raaed-api`) في بوابة Azure واضغط على زر **Stop**. لن تدفع تكاليف التشغيل عند توقف الخدمة. تذكر إعادة تشغيلها بالضغط على **Start** قبل البدء في التطوير أو العرض أمام اللجنة.
2. **استخدام طبقة B2/B3 المؤقتة:**
   يمكنك استخدام طبقة عالية الأداء أثناء العرض والتقييم لضمان سرعة فائقة للـ OCR ونماذج الذكاء الاصطناعي، ثم خفض تصنيف الخدمة (Scale down) إلى طبقة أقل بعد انتهاء العرض لتوفير المال.
3. **تنظيف السجل الخاص بالحاويات (ACR Cleanup):**
   عند بناء ورفع صور Docker متعددة، تستهلك الصور القديمة مساحات تخزين وتكلف مبالغ بسيطة. احرص دائمًا على مسح الإصدارات القديمة وإبقاء إصدار الإنتاج الفعلي فقط.

مبارك مقدمًا على هذا المشروع الرائع، ومستعد تمامًا لمساعدتك في أي خطوة إضافية لتسهيل رحلة الإطلاق والرفع السحابي! 🚀
