for ntest in 0 1 2 3 4 5 6 7 8 9
do
	python -u main.py  pLSA  samson   $ntest --seed $ntest
	python -u main.py  pLSA  jasper   $ntest --seed $ntest
	python -u main.py  pLSA  urban   $ntest --seed $ntest
	python -u main.py  pLSA  cuprite   $ntest --seed $ntest
done
for ntest in 0 1 2 3 4 5 6 7 8 9
do
	python -u main.py  dpLSA  samson   $ntest --dplsa_dim 250 --seed $ntest
	python -u main.py  dpLSA  jasper   $ntest --dplsa_dim 250 --seed $ntest
	python -u main.py  dpLSA  urban   $ntest --dplsa_dim 250 --seed $ntest
	python -u main.py  dpLSA  cuprite   $ntest --dplsa_dim 250 --seed $ntest
done


