library("readr")
library("dplyr")
library("plotly")
library("magrittr")
library("purrr")
library("tidyr")
library("writexl")

lambda_max <- 75
epsilon <- 1e-15
class_1_test_patterns <- c( 1:72 )
class_1_training_patterns <- c( 73:144 )
class_2_test_patterns <- c( 73:144 )
class_2_training_patterns <- c( 1:72 )
columns <- c( "mcv", "alkphos", "sgpt", "sgot", "gammagt", "drinks_num", "select")
features <- c( "alkphos", "sgpt", "gammagt")
#features <- c( "alkphos", "sgpt")

Bupa.Tib <- read_csv( "bupa.data", col_names = columns ) %>%
  tibble::rowid_to_column("id")

# Lambda == the Euclidian distance to the nearest observation 
# from the OTHER class
find_lambda <- function( observation, Other.Class.Tib, lambda_max, epsilon, features ) {
  Other.Class.Tib %>%
    select( features ) %>%
    mutate( euclid_dist = apply( . , 1, function(x) sqrt( sum( ( x - observation )^2 ) ) ) ) %>%
    select( euclid_dist ) %>%
    min() %>%
    min( . - epsilon, lambda_max ) }

rce_classify <- function( observation, Data.Tib, features ) {
  Data.Tib %>% 
    select( features ) %>%
    mutate( euclid_dist = apply( . , 1, function(x) sqrt( sum( ( x - observation )^2 ) ) ) ) %>%
    filter( euclid_dist < Data.Tib$lambda ) %>%
    nrow
}

rce_classify_tib <- function(Test.Data.Tib, Class.One.Train.Tib, Class.Two.Train.Tib, features) { Test.Data.Tib %<>%
  select( features ) %>%
  mutate( class.2.hits = apply( . , 1, function(x) rce_classify( x, Class.Two.Train.Tib , features ) ) ) %>%
    mutate( id = Test.Data.Tib$id )

Test.Data.Tib %<>%
  select(features) %>%
  mutate( class.1.hits = apply( . , 1, function(x) rce_classify( x, Class.One.Train.Tib, features ) ) ) %>%
  mutate( class.2.hits = Test.Data.Tib$class.2.hits,
          id = Test.Data.Tib$id ) %>%
  mutate( rce_class = ifelse( test = class.1.hits > class.2.hits,
                              yes = 1,
                              no = ifelse( test = class.2.hits > class.1.hits,
                                           yes = 2,
                                           no = 3))) 
  return(Test.Data.Tib)
}


# Class 1 Training patterns Tibble (Data Frame)
Class.1.Train.Tib <- Bupa.Tib %>% 
  filter( select == 1 ) %>%
  select( id, features) %>%
  slice( class_1_training_patterns )

# Class 2 Training patterns Tibble
Class.2.Train.Tib <- Bupa.Tib %>% 
  filter( select == 2 ) %>%
  select( id, features) %>%
  slice( class_2_training_patterns )

# Find Lambda for Class 1 Training patterns
Class.1.Train.Tib %<>% 
  select( features ) %>%
  mutate( lambda = apply(. , 1, function(x) find_lambda(x, 
                                                        Class.2.Train.Tib, 
                                                        lambda_max, 
                                                        epsilon,
                                                        features ) ) ) %>%
  mutate( id = Class.1.Train.Tib$id )

# Find Lambda for Class 1 Training patterns
Class.2.Train.Tib %<>% 
  select( features ) %>%
  mutate( lambda = apply(. , 1, function(x) find_lambda(x, 
                                                        Class.1.Train.Tib, 
                                                        lambda_max, 
                                                        epsilon,
                                                        features ) ) ) %>%
  mutate( id = Class.2.Train.Tib$id )


Test.Patterns <- Bupa.Tib %>%
  filter( select == 1 ) %>%
  slice( class_1_test_patterns ) %>%
  bind_rows( Bupa.Tib %>%
                filter( select == 2 ) %>%
                slice( class_2_test_patterns ) )

Test.Patterns %<>% rce_classify_tib(Class.1.Train.Tib, Class.2.Train.Tib, features) %>%
  left_join(Bupa.Tib, by = c(features, "id")) %>% 
  mutate( correct_class = select ) %>%
  select( id, features, class.1.hits, class.2.hits, correct_class, rce_class ) %>%
  mutate( error = ifelse( test = correct_class != rce_class,
                          yes = 1,
                          no = 0))

Test.Patterns %>% filter( rce_class != 3) %>% select(error) %>% sum/144

max_obs <- Class.1.Train.Tib %>% 
  bind_rows(Class.2.Train.Tib) %>% 
  select(features) %>%
  max

test_grid <- expand.grid( seq( 0, max_obs * 1.1, length.out = 50 ),
                          seq( 0, max_obs * 1.1, length.out = 50 ),
                          seq( 0, max_obs * 1.1, length.out = 50 ) ) 
names( test_grid ) <- features

test_grid %<>% 
  as_tibble %>% 
  tibble::rowid_to_column("id") %>% 
  rce_classify_tib( Class.1.Train.Tib, Class.2.Train.Tib, features) 

test_grid %>% filter( rce_class != 3 ) %>% 
  mutate( rce_class = ifelse( test = rce_class == 1,
                              yes  = "one",
                              no  = "two") ) %>%
  plot_ly( x = ~alkphos, y = ~sgpt, z = ~gammagt , color = ~rce_class )