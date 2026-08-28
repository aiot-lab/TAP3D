#!/bin/bash
echo "Downloading AnnotatedData..."
hf download TAP3DNow/TAP3D --include "AnnotatedData/*" --repo-type dataset --local-dir ./
# enter AnnotatedData/
cd ./AnnotatedData

# unzip all zip files to each folder
for zipfile in *.zip; do
    if [ -f "$zipfile" ]; then
        dirname="${zipfile%.zip}"
        mkdir -p "$dirname"
        unzip -q "$zipfile" -d "$dirname"
        rm -f "$zipfile"
        echo "✓ Extracted: $zipfile -> $dirname/"
    fi
done

for folder in ./*; do
    if [ -d "$folder" ]; then
        # check if source folder exists
        if [ -d "$folder/AnnotationUpload/$folder/senxor_m08" ]; then
            # build target folder
            mkdir -p "$folder/senxor_m08"
            # move contents
            mv "$folder/AnnotationUpload/$folder/senxor_m08"/* "$folder/senxor_m08/"
            # remove empty source folders
            rm -rf "$folder/AnnotationUpload/"
            echo "✓ Moved: $folder/AnnotationUpload/$folder/senxor_m08 -> $folder/senxor_m08"
        else
            echo "⚠ Warning: $folder/AnnotationUpload/$folder/senxor_m08 does not exist"
        fi
    fi
done

cd ..

echo "Downloading logs/"
hf download TAP3DNow/TAP3D logs.zip --repo-type dataset --local-dir ./
unzip logs.zip

echo "Downloading TAP3D_compare2others.zip/"
hf download TAP3DNow/TAP3D TAP3D_compare2others.zip --repo-type dataset --local-dir ./
unzip TAP3D_compare2others.zip

echo "Downloading weights.zip/"
hf download TAP3DNow/TAP3D weights.zip --repo-type dataset --local-dir ./
unzip weights.zip

echo "✅ Done!"
