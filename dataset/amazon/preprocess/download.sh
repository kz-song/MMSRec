# http://jmcauley.ucsd.edu/data/amazon/index_2014.html

cd ../raw

# Amazon All Beauty
mkdir Beauty
cd ./Beauty
wget --no-check-certificate http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Beauty.csv
wget --no-check-certificate http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz
cd ../

# Amazon Sports and Outdoors
mkdir Sports
cd ./Sports
wget --no-check-certificate http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Sports_and_Outdoors.csv
wget --no-check-certificate http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Sports_and_Outdoors.json.gz
cd ../

# Amazon Clothing Shoes and Jewelry
mkdir Clothing
cd ./Clothing
wget --no-check-certificate http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Clothing_Shoes_and_Jewelry.csv
wget --no-check-certificate http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Clothing_Shoes_and_Jewelry.json.gz
cd ../

# Amazon Home and Kitchen
mkdir Home
cd ./Home
wget --no-check-certificate http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Home_and_Kitchen.csv
wget --no-check-certificate http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Home_and_Kitchen.json.gz
cd ../




