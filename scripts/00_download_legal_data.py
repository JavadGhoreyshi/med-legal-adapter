import os
from datasets import load_dataset

TARGET_SIZE_MB = 60
TARGET_SIZE_BYTES = TARGET_SIZE_MB * 1024 * 1024
output_file = "legal_train_60mb.txt"

print(f"در حال اتصال استریم به PerSets/iran-legal-persian-qa و استخراج {TARGET_SIZE_MB} مگابایت...")

# استفاده از streaming=True برای جلوگیری از دانلود کامل دیتای حجیم
dataset = load_dataset(
    "PerSets/iran-legal-persian-qa", 
    split="train", 
    streaming=True
)

bytes_written = 0
records_count = 0

with open(output_file, "w", encoding="utf-8") as f:
    for item in dataset:
        # استخراج متون از کلیدهای مرسوم پرسش و پاسخ
        # در این دیتاست‌ها معمولاً کلیدها question/answer یا context یا text هستند
        q_text = item.get("question") or item.get("title") or ""
        a_text = item.get("answer") or item.get("response") or item.get("context") or ""
        
        # در صورتی که ساختار کلیدها متفاوت باشد، تمام مقادیر متنی را تجمیع می‌کنیم
        if not q_text and not a_text:
            text_chunks = [str(v) for v in item.values() if isinstance(v, str)]
            combined_line = " ".join(text_chunks).strip()
        else:
            combined_line = f"{q_text} {a_text}".strip()
            
        combined_line = combined_line.replace("\n", " ")
        
        if combined_line:
            line_to_write = combined_line + "\n"
            f.write(line_to_write)
            bytes_written += len(line_to_write.encode("utf-8"))
            records_count += 1
            
        # بررسی شرط توقف بر اساس حجم بایت‌های نوشته‌شده
        if bytes_written >= TARGET_SIZE_BYTES:
            print(f"\nبه سقف هدف {TARGET_SIZE_MB} مگابایت رسیدیم!")
            break

print(f"استخراج با موفقیت پایان یافت:")
print(f"- تعداد سوابق ثبت‌شده: {records_count:,}")
print(f"- حجم دقیق فایل: {bytes_written / (1024*1024):.2f} MB")
print(f"- مسیر فایل: {output_file}")
