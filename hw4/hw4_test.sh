#!/bin/bash 
wget -O w2v_Model_256 'https://www.dropbox.com/s/8edxvscp7k3ahrp/w2v_Model_256?dl=1'
python3 hw4_test.py $1 $2
