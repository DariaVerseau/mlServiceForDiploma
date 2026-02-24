# test_style.py
try:
    # Читаем изображения
    with open("cat.jpg", "rb") as f:
        content = f.read()
        
    with open("styles/erinHanson.jpg", "rb") as f:
        style = f.read()

    # Применяем стиль
    from basic_style_transfer import process_image
    result = process_image(content, style)

    # Сохраняем
    with open("result_middle_cat_eHanson.jpg", "wb") as f:
        f.write(result)
        
    print("✅ Успех! Результат сохранён как result.jpg")
    
except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()