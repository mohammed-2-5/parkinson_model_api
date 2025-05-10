# استخدام صورة Python الأساسية
FROM python:3.10-slim

# تحديد مجلد العمل داخل الحاوية
WORKDIR /app

# نسخ جميع ملفات المشروع إلى الحاوية
COPY . .

# تثبيت الحزم المطلوبة
RUN pip install --no-cache-dir -r requirements.txt

# تعيين المنفذ الذي سيعمل عليه التطبيق
EXPOSE 8080

# أمر تشغيل التطبيق باستخدام Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
