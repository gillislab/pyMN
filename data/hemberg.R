library(scRNAseq) 

my_data <- list(
	baron = BaronPancreasData(),
	lawlor = LawlorPancreasData(),
	seger = SegerstolpePancreasData(),
	muraro = MuraroPancreasData()
)
 
rownames(my_data$muraro) <- rowData(my_data$muraro)$symbol 
my_data$muraro <- my_data$muraro[!duplicated(rownames(my_data$muraro)),]

library(org.Hs.eg.db)
symbols<-mapIds(org.Hs.eg.db, keys=rownames(my_data$lawlor), keytype='ENSEMBL',column='SYMBOL')

keep <-!is.na(symbols)& !duplicated(symbols)
my_data$lawlor<-my_data$lawlor[keep,]
rownames(my_data$lawlor) <- symbols[keep]

my_data$baron$"cell type" <-my_data$baron$label
my_data$muraro$"cell type" <-my_data$muraro$label

fused_data<-MetaNeighbor::mergeSCE(my_data)


Matrix::writeMM(counts(fused_data),'pancreas.mtx')
write.csv(colData(fused_data),'pancreas_col.csv',quote=F)
write.table(rownames(fused_data),'pancreas_genes.csv',quote=F,row.names=F,col.names=F)