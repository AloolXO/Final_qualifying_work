# Final_qualifying_work

## Описание

Программа предназначена для работы со спутниковыми снимками, включающая предобработку данных, конвертацию форматов, кадрирование изображений и взаимодействие с архитектурой Segformer для выделения объектов горной промышленности.

## Функции

- **Предобработка**: Очистка и нормализация спутниковых снимков.
- **Конвертация**: Перевод изображений из одного формата в другой.
- **Кадрирование**: Извлечение заданных областей изображений.
- **Segformer**: Интеграция с архитектурой Segformer для сегментации объектов горной промышленности.

## Требования

- Python 3.11
- Указанные в `requirements.txt` библиотеки

## Установка

1. Запустите установочный файл `FinalQualifyingWorkSetup.exe`.
2. Следуйте инструкциям установщика для установки Python и всех необходимых библиотек.
3. Запустите программу через созданный ярлык или напрямую через файл `Final_qualifying_work.py`.

## Использование

1. **Предобработка**:
   ```python
   from preprocessing import preprocess_image
   preprocess_image('input_image.jpg', 'output_image.jpg')
   ```

2. **Конвертация**:
   ```python
   from conversion import convert_format
   convert_format('input_image.jpg', 'output_image.png')
   ```

3. **Кадрирование**:
   ```python
   from cropping import crop_image
   crop_image('input_image.jpg', 'output_image.jpg', (x, y, width, height))
   ```

4. **Segformer**:
   ```python
   from segformer import segment_image
   segment_image('input_image.jpg', 'output_segmented.jpg')
   ```

## Лицензия

Этот проект лицензирован на условиях лицензии MIT. Подробнее см. [LICENSE](LICENSE.txt).

## Контакты

[Leunenko Artem Olegovich]  
[al.university.edu@yandex.ru]