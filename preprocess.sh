TEMP_FILE=all_files.txt

IN_FOLDER=$1
if [ -z "$1" ]
  then
    echo "Error: must specifiy folder with mesh (*.obj) video"
    exit -1
fi

OUT_FOLDER=$2
if [ -z "$2" ]
  then
    echo "Error: must specifiy out folder for (*.npz)"
    exit -1
fi

find $IN_FOLDER -type f -name "*.obj" > $TEMP_FILE
while read filepath; do
    filename=$(basename $filepath)
    filename="${filename%.*}"
    ./DeepSDF/bin/PreprocessMesh -m $filepath -o "$OUT_FOLDER/$filename.npz" -t &
    [ $( jobs | wc -l ) -ge $( nproc ) ] && wait & echo "Done $filepath"
done < $TEMP_FILE
