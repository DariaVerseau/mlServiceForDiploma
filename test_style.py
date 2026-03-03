# test_style.py
try:
    # Читаем изображения
    with open("test_photo_originals/morning_road.JPG", "rb") as f:
        content = f.read()
        
    with open("styles/monet2.jpg", "rb") as f:
        style = f.read()

    # Применяем стиль
    from basic_style_transfer import process_image
    result = process_image(content, style)

    # Добавь .jpg в конец имени файла
    with open("results/result_morning_road_monet2.jpg", "wb") as f:
        f.write(result)
        
    print("✅ Успех! Результат сохранён как result_morning_road_monet2.jpg")
    
except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()