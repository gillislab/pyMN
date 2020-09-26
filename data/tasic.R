library(scRNAseq)
tasic = TasicBrainData(ensembl = FALSE)
tasic$study_id <- 'tasic'

Matrix::writeMM(Matrix::Matrix(counts(tasic),sparse=T),'tasic_counts.mtx')
write.table(rownames(tasic), 'tasic_genes.csv',row.names = F,col.names = F,quote=F)
write.csv(colData(tasic)[,c('study_id','primary_type')],'tasic_col.csv',quote = F)

