pip install -r requirements.txt
mkdir -p logs
python q1.py | tee logs/q1.log
python q2.py | tee logs/q2.log
python q3.py | tee logs/q3.log
