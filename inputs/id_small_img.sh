find . -iname "*.jpg" -type f -exec identify -format '%w %h %i\n' '{}' \; | awk '$1<128 || $2<128'
