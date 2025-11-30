#--- ANÁLISIS DISCRIMINANTE (LDA) ---
# Equipo: Diego Ceciliano Díaz (A01424408), Rania Maya Suazo (A01067915), Laura Valeria Gutierrez Rangel (A01424392)
# 13 de noviembre del 2025

setwd("~/Downloads/Multivariante/Proyecto/")

# 0) Paquetes necesarios
# install.packages(c("MASS","ggplot2","caret","readxl","dplyr","biotools")))
library(MASS) # lda()
library(ggplot2) # gráficos
library(caret) # partición train/test y métricas
library(readxl) # leer base de datos
library(tidyr)  # drop_na()
library(dplyr) # manipulación de datos
library(biotools)

# 1) Cargar datos
datos <- readxl::read_excel("df_pob_ent2.xlsx")

# Verificar años disponibles
table(datos$anio)

# Crear lista de años
anios <- c(2016, 2022)

# Crear lista para almacenar modelos y resultados
resultados <- list()


# ============================================================
# *** INICIO DEL LOOP: análisis para 2016 y 2022 ***
# ============================================================

for (a in anios) {
  
  cat("\n==============================\n")
  cat("  ANALISIS PARA EL AÑO:", a, "\n")
  cat("==============================\n\n")
  
  # Filtrar año
  datos_a <- datos %>% filter(anio == a)
  
  # 2) Crear variable categórica de tipo de vulnerabilidad con 3 grupos
  med_car <- median(datos_a$vul_car, na.rm = TRUE)
  med_ing <- median(datos_a$vul_ing, na.rm = TRUE)
  
  datos_a <- datos_a %>%
    mutate(
      tipo_vulnerabilidad = case_when(
        vul_car >= med_car ~ "Vulnerable_Carencias",
        vul_ing >= med_ing ~ "Vulnerable_Ingreso",
        TRUE ~ "no_pobv"
      )
    )
  
  cat("\nConteo por grupo en", a, ":\n")
  print(table(datos_a$tipo_vulnerabilidad))
  
  
  # 3) Seleccionar variables explicativas y limpiar NA
  vars_exp <- c("ic_rezedu","ic_asalud","ic_segsoc","ic_cv","ic_sbv","ic_ali_nc")
  
  datos_a$tipo_vulnerabilidad <- factor(
    datos_a$tipo_vulnerabilidad,
    levels = c("no_pobv","Vulnerable_Carencias","Vulnerable_Ingreso")
  )
  
  datos_clean <- datos_a %>% drop_na(all_of(vars_exp), tipo_vulnerabilidad)
  
  cat("\nTamaño por grupo tras limpieza en", a, ":\n")
  print(table(datos_clean$tipo_vulnerabilidad))
  
  
  # 4) Dividir en entrenamiento y prueba
  set.seed(123)
  idx <- createDataPartition(
    datos_clean$tipo_vulnerabilidad, 
    p = 0.7, 
    list = FALSE
  )
  train <- datos_clean[idx, ]
  test <- datos_clean[-idx, ]
  
  
  # 5) Verificación de supuestos 
  normalidad <- by(
    train[, vars_exp],
    train$tipo_vulnerabilidad,
    function(x) apply(x, 2, function(z) 
      if(length(z)>=3) shapiro.test(z)$p.value else NA)
  )
  
  cat("\nNormalidad Shapiro-Wilk en", a, ":\n")
  print(normalidad)
  
  cat("\nBox's M en", a, ":\n")
  boxm_res <- biotools::boxM(train[, vars_exp], train$tipo_vulnerabilidad)
  print(boxm_res)
  
  
  # 6) Ajustar modelo LDA
  modelo_lda <- MASS::lda(
    tipo_vulnerabilidad ~ ic_rezedu + ic_asalud + ic_segsoc + ic_cv + ic_sbv + ic_ali_nc,
    data = train
  )
  
  print(modelo_lda)
  
  
  # 7) Extraer scalings
  coef_df <- as.data.frame(modelo_lda$scaling)
  coef_df$Indicador <- rownames(coef_df)
  coef_df$Abs_LD1 <- abs(coef_df$LD1)
  coef_df$Abs_LD2 <- abs(coef_df$LD2)
  
  rank_LD1 <- coef_df[order(-coef_df$Abs_LD1),
                      c("Indicador","LD1","Abs_LD1")]
  rank_LD2 <- coef_df[order(-coef_df$Abs_LD2),
                      c("Indicador","LD2","Abs_LD2")]
  
  cat("\nRanking LD1 en", a, ":\n")
  print(rank_LD1)
  cat("\nRanking LD2 en", a, ":\n")
  print(rank_LD2)
  
  
  # 8) Predicción en test
  pred_test <- predict(modelo_lda, newdata = test)
  conf_mat <- table(Real = test$tipo_vulnerabilidad, Predicho = pred_test$class)
  
  cat("\nMatriz de confusión en", a, ":\n")
  print(conf_mat)
  
  accuracy <- mean(pred_test$class == test$tipo_vulnerabilidad)
  cat("Precisión (test) en", a, ":", round(accuracy*100,2), "%\n")
  
  
  # 9) Visualización LD1 vs LD2
  train_scores <- predict(modelo_lda)$x
  viz <- data.frame(
    LD1 = train_scores[,1],
    LD2 = if(ncol(train_scores)>=2) train_scores[,2] else 0,
    Clase = train$tipo_vulnerabilidad
  )
  
  p <- ggplot(viz, aes(x = LD1, y = LD2, color = Clase)) +
    geom_point(size = 3, alpha = 0.8) +
    stat_ellipse(level = 0.68, linetype = "dashed") +
    theme_minimal(base_size = 13) +
    labs(
      title = paste("Separación entre grupos (LDA) - Año", a),
      x = "Función discriminante 1 (LD1)",
      y = "Función discriminante 2 (LD2)"
    )
  
  print(p)

  
  # 10) Validación cruzada LOOCV
  modelo_cv <- lda(
    tipo_vulnerabilidad ~ ic_rezedu + ic_asalud + ic_segsoc + ic_cv + ic_sbv + ic_ali_nc,
    data = datos_clean,
    CV = TRUE
  )
  
  cv_acc <- mean(modelo_cv$class == datos_clean$tipo_vulnerabilidad)
  cat("Precisión CV (LOOCV) en", a, ":", round(cv_acc*100,2), "%\n")
  
  
  # Guardar resultados en lista
  resultados[[as.character(a)]] <- list(
    normalidad = normalidad,
    boxM = boxm_res,
    modelo = modelo_lda,
    scalings = coef_df,
    accuracy = accuracy,
    accuracy_cv = cv_acc,
    conf_mat = conf_mat,
    grafica = p
  )
  
}

# ============================================================
# FIN DEL LOOP
# ============================================================

cat("\n\n==== RESUMEN FINAL DE RESULTADOS POR AÑO ====\n")
print(resultados)

