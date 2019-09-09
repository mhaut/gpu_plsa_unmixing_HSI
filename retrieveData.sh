#!/bin/bash

if [ -d "inputs" ]; then
    read -p "Do you wish to overwrite inputs folder (CAUTION: ACTUAL FOLDER WILL BE DELETED)?" yn
    case $yn in
        [Yy]* ) rm -r inputs;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no."; exit;;
    esac
fi

if [ -d "groundtruths" ]; then
    rm -r groundtruths
fi

mkdir inputs
mkdir groundtruths

echo 'Retrieving Samson...'
wget http://www.escience.cn/system/file?fileId=68596 -O 'inputs/samson.zip'
wget http://www.escience.cn/system/file?fileId=69115 -O 'inputs/samson_gt.zip'
unzip -qq inputs/samson.zip
mv Data_Envi/* inputs/
rm -r Data_Envi

unzip -qq inputs/samson_gt.zip
mv GroundTruth/end3.mat groundtruths/samson_gt.mat
rm -r GroundTruth/

echo 'Retrieving Jasper...'
wget http://www.escience.cn/system/file?fileId=69113 -O 'inputs/jasperRidge2_R198.mat'
wget http://www.escience.cn/system/file?fileId=69114 -O 'inputs/jasper_gt.zip'
unzip -qq inputs/jasper_gt.zip
mv GroundTruth/end4.mat groundtruths/jasper_gt.mat
rm -r GroundTruth/

echo 'Retrieving Urban...'
wget http://www.escience.cn/system/file?fileId=69117 -O 'inputs/Urban_R162.mat'
wget http://www.escience.cn/system/file?fileId=69120 -O 'inputs/urban_gt.zip'
unzip -qq inputs/urban_gt.zip
mv groundTruth/end4_groundTruth.mat groundtruths/urban_gt.mat
rm -r groundTruth/

echo 'Retrieving Cuprite...'
wget http://www.escience.cn/system/file?fileId=69134 -O 'inputs/Cuprite_S1_R188.mat'
wget http://www.escience.cn/system/file?fileId=69129 -O 'inputs/cuprite_gt.zip'
unzip -qq inputs/cuprite_gt.zip
mv groundTruth_Cuprite_end12/groundTruth_Cuprite_end12 groundtruths/cuprite_gt.mat

rm -r inputs/*.zip
