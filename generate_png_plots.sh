python generate_png_plots.py
for filename in ../reports/ifac-raw-plots/*.png; do
    convert -trim $filename $filename
done
$SHELL