for f in mmd_no_sigmoid/csv/*train.csv
do
f1=${${f/*csv\//}/_train\.csv/}
f2=mmd_no_sigmoid/${f1}_fit.pkl
python inference.py --data-dir='mmd_no_sigmoid/csv/' --data-name=$f1 -v --stan-model='model.stan' --model-path='mmd_no_sigmoid/model.pkl' --fit-save-path=$f2 --vb
done
