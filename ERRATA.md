# Fe de erratas

Trabajo Fin de Grado: «Clasificación automática de lesiones cutáneas mediante redes neuronales convolucionales»  
Daniel Ortiz Requena. Grado en Ingeniería de la Salud, Universidad de Málaga  
Septiembre de 2026  

Después de depositar la memoria encontré varios errores. En este documento recojo qué se ha corregido y dónde, ordenado por capítulos y secciones.

Las correcciones son de redacción, de datos y referencias de trabajos de otros autores, y de la descripción de algunos detalles del código. Los resultados del trabajo no cambian: las tablas de resultados, las figuras y las cifras obtenidas son las mismas que en la versión depositada. El código y los resultados publicados en el repositorio del proyecto (etiqueta v1.0) tampoco cambian.

## Agradecimientos

- Errata tipográfica: donde ponía «forma parte en la vida» pone ahora «formar parte de la vida».

## Resumen y Abstract

- Se decía que la calibración elevó la precisión balanceada del subgrupo maligno hasta 0,804 de media en las tres semillas. La subida solo está medida en la ejecución de referencia; en las tres semillas lo comprobado es el valor que se alcanza tras calibrar. Ahora dice que ese valor se situó en 0,804 de media. El Abstract se corrige igual.

## Capítulo 1

- 1.4. Se decía que el conjunto de test se mantuvo aislado durante todas las decisiones de diseño. Ahora dice que no interviene en la calibración, que es lo que se puede afirmar. En la misma sección se elimina la palabra «favorable» en la frase sobre la ejecución única.
- 1.5. Se quita la palabra «cerrado» al referirse al conjunto de test oficial del reto, porque sus etiquetas son públicas.

## Capítulo 2

- 2.3.2. Se atribuía a He et al. que el problema de las redes muy profundas era el desvanecimiento del gradiente. Ellos lo describen como un problema de degradación del entrenamiento y descartan que se deba al gradiente. Se corrige. En 2.3.2 y 2.3.3 se añade un punto final que faltaba.
- 2.4. La frase sobre combinar estrategias ponía como ejemplo el WeightedRandomSampler y la augmentación, que actúan los dos sobre los datos. Ahora dice que este trabajo los combina con la Focal Loss, que actúa sobre la función de pérdida.
- 2.5. Se elimina la explicación de por qué el TTA no usa cambios de color. Decía que alterarían la calibración, pero la calibración ya se hace sobre validación con el mismo TTA, así que el argumento no era correcto.
- 2.6. El criterio F-beta aparecía como una media entre precisión y recall. En el código combina sensibilidad y especificidad. En la fuente de la Tabla 2.1 se añade ejecucion_log.txt, que es el fichero del que salen las métricas de validación.
- 2.8. Se corrigen los datos de otros trabajos. Gessert et al. obtuvieron 0,856 (ponía 0,860) y Mahbod et al. 0,862 (ponía 0,837), y los dos usan datos externos. El 0,872 de Kitada e Iyatomi está medido sobre el conjunto de validación del reto y no sobre test. La cita de Gessert et al. aparecía como [8] y es la [9]. El ganador del reto obtuvo 0,885 (ponía que lo superó) y se quita que sus datos externos no fueran públicos, porque la fuente no lo dice. Con estos datos, el resultado de este trabajo se describe ahora como situado en el rango de los métodos que no usan datos externos ni metadatos.
- 2.9. Se decía que la mejora del subgrupo maligno se repetía en las tres semillas. Esa mejora solo está medida en la ejecución de referencia. Ahora dice que lo estable entre semillas es el punto de operación que se alcanza con la calibración.

## Capítulo 3

- 3.3. Los nombres de los niveles de augmentation no eran los del código. Ahora se usan los del código (leve, media y agresiva) y el nivel 0 aparece como «sin augmentación».
- 3.4.2. Se quita que la proporción 60/20/20 sea «estándar», porque la referencia de Kohavi que acompañaba a la frase no lo dice. La frase sobre el aislamiento del test decía que el test no se usa en ninguna decisión de diseño. Ahora dice que no interviene en ninguna decisión dentro de cada ejecución, es decir, ni en el early stopping ni en los pesos del ensemble ni en los umbrales.
- 3.5. Se decía que las imágenes de HAM10000 se recogieron con consentimiento informado, que las validaron dermatólogos certificados y que todas se tomaron con dermatoscopios calibrados. El artículo del dataset no dice eso. Ahora se indica lo que sí recoge: la aprobación de los comités de ética de Viena y Queensland y la anonimización de las imágenes. También se añade la licencia del dataset (CC BY-NC 4.0), que hay que indicar al reproducir sus imágenes.

## Capítulo 4

- 4.1. La aportación de cada etapa en la Tabla 5.2 se mide de forma acumulativa; el texto decía «aislada». La calibración de umbrales se aplica a MEL y AKIEC, y el texto decía que a cada clase maligna. El ±0,009 se llamaba varianza y ahora se llama dispersión.
- 4.2.1. La evaluación determinista se refiere a cada modelo por separado, porque el TTA sí tiene una parte aleatoria.
- 4.2.2. La descripción de las transformaciones de los niveles 2 y 3 no coincidía con el código y se corrige. La sensibilidad de MEL en la versión previa del sistema era 0,682 (ponía «en torno a 0,65»), y en esa versión faltaban tanto el boost como la Focal Loss. Se quita también la justificación del color en el TTA, por el mismo motivo que en 2.5.
- 4.3.1. La cita de Esteva et al., que no usa ninguna de las tres arquitecturas, se cambia por Gessert et al. y Mahbod et al. ResNet-50 aparecía como el modelo más robusto: es el mejor en la ejecución de referencia, pero no el más estable entre semillas. La descripción de ResNet se ajusta a lo que dicen He et al., igual que en 2.3.2, y se añade un punto que faltaba.
- 4.3.2. Se quita que igualar el weight decay al learning rate mantenga la misma regularización en los tres modelos.
- 4.3.3. Se eliminan afirmaciones que se atribuían a trabajos que no las contienen: un rango numérico de learning rate atribuido a Tajbakhsh et al., unas explicaciones sobre el gradiente atribuidas a Huang et al. y a Tan y Le, y una relación entre weight decay y learning rate atribuida a Loshchilov y Hutter. Queda lo que esos trabajos sí dicen y lo observado en las ejecuciones. Se elimina también una prueba con weight decay uniforme que no llegó a hacerse. La referencia a las figuras de las curvas de entrenamiento pasa a ser a las Figuras 5.7, 5.8 y 5.9.
- 4.6.2. Ahora dice que el test no participa en la ponderación del ensemble. Antes decía que no participaba en ninguna decisión de diseño.
- 4.7.2. Se compararon 5 y 10 rondas de TTA; el texto decía que se habían probado más de 10. La consecuencia para la calibración queda explicada ahora como lo que es: validación y test tienen que usar el mismo TTA.
- 4.8.2. La calibración determina un umbral para MEL y AKIEC. El texto decía que para cada clase maligna.
- 4.9.2. La capa de Grad-CAM de EfficientNet-B3 es features[8][0].
- 4.10. El orden de detección del entorno era el contrario al del código, que comprueba primero Kaggle. Los tests están en seis suites y no hay ninguna de forward pass de las redes. Los tests tampoco cubren la detección del entorno. El fichero split_assignment.json no está en el repositorio, lo genera el código. Se quita «sin riesgo de data leakage», porque HAM10000 tiene varias imágenes de algunas lesiones y la partición se hace por imagen.

## Capítulo 5

- 5.3. Igual que en 4.1, la contribución de cada etapa se mide de forma acumulativa.
- 5.4. Las clases con especificidad igual o superior a 0,98 son las distintas de NV y MEL; el texto decía las distintas de NV. BCC no tiene un umbral clínico definido, así que ahora solo se dice que su sensibilidad es la más alta de las malignas. MEL y AKIEC se presentaban como las clases con menos datos, y MEL no lo es; ahora se dice que son las dos con objetivo clínico. Los 24 falsos negativos de AKIEC suponen omitir casi un tercio de la clase, y el texto hablaba de una diferencia de un tercio respecto al objetivo.
- 5.4.1. AKIEC y BCC son lesiones queratinocíticas (ponía «queratinizantes»). El subtipo de BKL es la queratosis liquenoide de tipo liquen plano (ponía «liquen plano actínico»). Se quita la mención a los glóbulos azul-grisáceos del carcinoma basocelular, porque la nueva referencia [25] no trata ese rasgo. DF pierde 5 de sus 32 imágenes, así que su confusión no es prácticamente nula; VASC no pierde ninguna.
- 5.5. Cada semilla genera su propia partición con el mismo procedimiento. El texto decía que todas usaban el mismo split. La desviación típica de 0,009 (ponía «varianza») tiene varias fuentes que en este diseño no se pueden separar: la propia semilla, el no determinismo de la GPU y el uso de dos plataformas. También se corrigen dos frases que daban la mejora del subgrupo maligno como repetida en las tres semillas.
- 5.6.1. Se corrigen las explicaciones de la precisión media de DF y VASC. En VASC el orden de las probabilidades no es perfecto, tiene dos excepciones, y el AP es 0,995.
- 5.7. La pérdida de validación baja hasta las épocas 20 a 23 (ponía 30 a 35) y la BACC de entrenamiento supera a la de validación desde la primera época (ponía «a partir de las épocas intermedias»).
- 5.8 y Tabla 5.5. Se corrigen los valores de Gessert et al. y Mahbod et al. y su uso de datos externos. El 0,887 es de Sun et al.; se atribuía a Liu et al. Se añade una columna con el conjunto sobre el que se evalúa cada método, porque la cifra de Kitada e Iyatomi es de validación. Se quita que el test oficial no sea público y se explica qué se entiende por datos externos, siguiendo a Shen et al. El texto de la sección se ajusta a la nueva tabla: MetaOptima ya no la encabeza y se habla de entradas del reto en lugar de métodos ganadores.
- 5.9. La referencia [31] es la regla ABCD de dermatoscopia; ponía criterios ABCDE. Se quita la descripción del mapa de DF, que no coincidía con la figura.

## Capítulo 6

- 6.1. La ganancia de 0,027 está medida en la ejecución de referencia. Lo que se mantiene en las tres semillas es el valor calibrado, entre 0,801 y 0,806.
- 6.2. La mejor época de EfficientNet-B3 es la 6; el texto decía que el entrenamiento terminaba en la 6, y termina en la 16.
- 6.3. En los experimentos previos solo se probó F-beta. La calibración de temperatura y la de Platt no se llegaron a probar, aunque el texto decía que sí. F-beta subía la sensibilidad de MEL a costa de bajar la de NV.
- 6.4.1. La dispersión de AKIEC entre semillas se debe a la variación de semilla, partición y plataforma. ISIC 2020 no tiene queratosis actínica, así que solo se menciona ISIC 2019.
- 6.4.2. Lo mismo que en 5.5 sobre las fuentes de la dispersión y el uso de «desviación típica».
- 6.5. Con los valores corregidos, el sistema no supera a Mahbod et al.: queda a 0,016 de ellos y a un punto de Gessert et al., que usan datos externos. Se añade que la comparación no está equiparada porque las particiones son distintas. La frase sobre los métodos que superan 0,870 se refiere ahora a los evaluados sobre el test oficial, porque Kitada e Iyatomi superan 0,870 sin datos externos. La frase que contaba esos recursos decía «tres de los cuatro» y ahora dice «dos de los tres». Se elimina una estimación de ganancia (entre 0,010 y 0,015) que no salía de ningún experimento.
- 6.6. Solo se menciona ISIC 2019 y se actualiza la numeración de las referencias.

## Capítulo 7

- 7.1. La sensibilidad de MEL de la versión previa era 0,682 y el valor final incluye también el efecto de la calibración. La mejora de 0,027 es de la ejecución de referencia; entre semillas lo estable es el valor calibrado (0,804 ± 0,002). Donde ponía varianza, ahora pone desviación típica.
- 7.3. El mismo ajuste sobre la mejora y el valor calibrado.
- 7.4. Solo se menciona ISIC 2019. Los metadatos del paciente los usan algunos de los métodos que superan 0,870; el texto decía que todos de forma sistemática.
- 7.5 y 7.6. La frase sobre la semilla favorable se refiere ahora al nivel alcanzado tras la calibración.

## Apéndices

- Apéndice A. Se quita que el split serializado esté en el repositorio. Se corrigen el nombre environment.yml y la versión de seaborn (0.12). La Tabla A.2 recoge ahora las seis suites de tests que hay en el repositorio, que suman los mismos 46 tests.
- Apéndice B. Se corrigen los nombres de los ficheros de resultados y se indica dónde están en el repositorio. La leyenda de la Tabla B.1 aparecía como «Tabla B0.1». En B.2, lo que se da por estable entre semillas es el punto de operación que produce la calibración, y las semillas en las que la calibración baja un poco la BACC global son la 7 y la 123; el texto solo mencionaba la 7. En B.3, AKIEC es la clase con más variabilidad entre las malignas; el texto decía que entre todas. En B.4 se aclara que la ganancia respecto al argmax solo está medida en la ejecución de referencia.

## Referencias

- [9] Gessert et al.: la entrada correcta es el preprint del reto ISIC 2018 (arXiv:1808.01694, 2018). La versión depositada citaba un artículo de 2020 que corresponde al reto de 2019.
- [25] y [26] no respaldaban lo que se citaba con ellas. Se sustituyen por Zalaudek et al. (2010), sobre estructuras vasculares en tumores no melanocíticos, y Elgart (2001), sobre queratosis seborreica, lentigo solar y queratosis liquenoide.
- [28] Sun et al. (2021) sustituye a Liu et al. (2020) como fuente del 0,887.
- [30] Shen et al. (2022) es una referencia nueva.
- Con estos cambios la lista pasa de 35 a 36 referencias y se renumeran desde la 28.
