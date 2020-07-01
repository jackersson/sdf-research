TEMP_FILE=all_files.txt

IN_FOLDER=/home/taras/coder/projects/sdf-research/data/planes_sample
OUT_FOLDER=/home/taras/coder/projects/sdf-research/sdf/parallel

find $IN_FOLDER -type f -name "*.obj" > $TEMP_FILE
while read filepath; do
    filename=$(basename $filepath)
    filename="${filename%.*}"
    ./DeepSDF/bin/PreprocessMesh -m $filepath -o "$OUT_FOLDER/$filename.npz" -t &
    [ $( jobs | wc -l ) -ge $( nproc ) ] && wait & echo "Done $filepath"
    # echo "Done $filepath"
done < $TEMP_FILE
