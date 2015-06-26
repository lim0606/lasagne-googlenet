#!/bin/bash

filenames=$(ls)
#echo $filenames
for filename in $filenames
do
  #echo $filename
  if [[ $filename == *".pkl" ]]
  then
    #echo "asdf"
    arr=$(echo $filename | tr "." "\n")
    i=0
    for x in $arr
    do
      #echo $x
      if [[ $i == 0 ]]  
      then
        aa=$x".weight.pkl"
        #echo $aa
        mv $filename $aa
      fi 
      echo "wtf $((i++))"
    done
     
    #mv $filename
  fi   
done
