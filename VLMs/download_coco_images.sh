mkdir images
cd images
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
mv val2017/*.jpg ./
rm val2017.zip
rm -r val2017