#!/bin/bash

for file in *.xml
do
    echo '<annotation>' > /tmp/$file
    cat $file >> /tmp/$file
    cp /tmp/$file $file
done
done
