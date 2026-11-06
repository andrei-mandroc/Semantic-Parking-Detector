# Annotation visualization

Run from the project root with a matching PKLot image and XML file:

```powershell
.\.venv\Scripts\python.exe -m scripts.visualize_annotations --image data/raw/PKLot/2013-04-14/2013-04-14_06_40_00.jpg --xml data/raw/PKLot/2013-04-14/2013-04-14_06_40_00.xml --mode geometry
```

Use `--mode ground-truth` to color free spaces green and occupied spaces red according to the XML labels. Spaces without a valid label are gray. This mode shows reference annotations, not AI predictions. The default `geometry` mode shows only contours and IDs in cyan. Images are saved in `outputs/`; use `--output path/to/image.png` to choose another file.
