training_data <- read.csv(file="/Codebase/Github/accord-census-classifier/data/adult.data", head=FALSE)
summary(training_data)
training_data_copy <- training_data
sapply(training_data_copy, function(x) if("factor" %in% class(x)) { as.numeric(x) } else { x })
training_data_converted <-
sapply(training_data_copy, function(x) if("factor" %in% class(x)) { as.numeric(x) } else { x })
)
write.table(training_data_converted, file="/Codebase/Github/accord-census-classifier/data/adult.converted.names", row.names=FALSE, col.names=FALSE, sep=",")
savehistory("C:/Codebase/Github/accord-census-classifier/data/data_converter.Rhistory")
