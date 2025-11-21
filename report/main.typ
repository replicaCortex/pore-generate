#set document(
  title: "Метод генерации статистически достоверных синтетических изображений пористых материалов для обучения нейросетевых моделей сегментации",
)
// #set heading(numbering: "1.")
#set text(lang: "ru", size: 14pt)
#set par(justify: true)

#set figure.caption(separator: [ --- ])

от объема и качества обучающих данных @Guo2022DeepLearning. Собрать
статистически достоверные структуры @Smith2025OnStatistical.

```json
{
  "image_settings": {
    "width": 200,
    "height": 200,
    "total_images": 10
  },
  "pore_settings": {
    "large_pores": {
      "count_range": [3, 5],
      "radius_mean": 25,
      "stretch_factor_range": [1, 1.1]
    },
    "medium_pores": {
      "count_range": [10, 15],
      "radius_mean": 12,
      "stretch_factor_range": [1, 1.1]
    },
    "small_pores": {
      "count_range": [20, 30],
      "radius_mean": 6
    }
  },
  "noise_settings": {
    "matrix_gray_range": [100, 200],
    "pore_gray_range": [0, 100],
    "noise_intensity": 0.1
  }
}
```,

```json
{
  "large_pores": [
    {
      "bbox": {"x_min": 50, "y_min": 40, "x_max": 88, "y_max": 82},
      "original": {
        "radius": 28,
        "area": 2463
      },
      "deformed": {
        "radius": 27,
        "area": 2335,
        "eccentricity": 0.2469,
        "circularity": 0.7408
      }
    }
  ]
}
```,


#bibliography(
  "bib.bib",
  title: "Список используемых источников",
  style: "gost-r-705-2008-numeric",
)
