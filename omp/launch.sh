#!/bin/bash -x  

for ntest in 0 1 2 
do
	./plsa_optimized.bin    samson pLSA  $ntest 
	./plsa_optimized.bin  jasper pLSA  $ntest 
	./plsa_optimized.bin    urban  pLSA $ntest 
	./plsa_optimized.bin   cuprite pLSA  $ntest 
done



for ntest in 0 1 2 
do
	./plsa.bin   samson pLSA  $ntest 
	./plsa.bin  jasper  pLSA $ntest 
	./plsa.bin    urban pLSA  $ntest 
	./plsa.bin   cuprite pLSA  $ntest 
done


for ntest in 0 1 2 
do
	./plsa_optimized.bin    samson dpLSA  $ntest 
	./plsa_optimized.bin  jasper dpLSA  $ntest 
	./plsa_optimized.bin    urban  dpLSA $ntest 
	./plsa_optimized.bin   cuprite dpLSA  $ntest 
done



for ntest in 0 1 2 
do
	./plsa.bin   samson dpLSA  $ntest 
	./plsa.bin  jasper  dpLSA $ntest 
	./plsa.bin    urban dpLSA  $ntest 
	./plsa.bin   cuprite dpLSA  $ntest 
done




