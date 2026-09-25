# Med-Legal XLM-RoBERTa Adapter

## وضعیت فعلی
- [x] Phase 0: استخراج توکن‌های تخصصی از corpus پزشکی+حقوقی
- [x] Phase 1: Warm-up embedding (فریز کامل بدنه، فقط embedding آزاد)
      → ذخیره‌شده در: shokol1011/med-legal-xlm-roberta
- [ ] Phase 2: باز کردن ۲ لایه‌ی آخر ترنسفورمر با lr پایین‌تر
- [ ] Phase 3: ارزیابی و/یا fine-tune روی وظیفه‌ی نهایی

## نحوه‌ی اجرا در Kaggle
1. Internet را در Settings نوت‌بوک روشن کن
2. `!git clone https://github.com/<username>/med-legal-adapter.git`
3. `%cd med-legal-adapter && pip install -r requirements.txt`
4. اسکریپت مرحله‌ی موردنظر را اجرا کن، مثلاً:
   `!python scripts/03_train_phase2_unfreeze.py`
