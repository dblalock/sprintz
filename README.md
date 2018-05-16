
Sprintz is a compression algorithm for multivariate integer time series. It requires only a few bytes of memory per variable, offers state-of-the-art compression ratios, and can decompress at multiple GB/s in a single thread.

See the [Sprintz paper](https://github.com/dblalock/sprintz/blob/master/assets/sprintz.pdf?raw=true) for details.

# Reproduction of Results

To reproduce any of the results in the paper, you can do the following.

## Install Dependencies

- [Joblib](https://github.com/joblib/joblib) - for caching function output
- [Pandas](http://pandas.pydata.org) - for storing results and reading in data
- [Seaborn](https://github.com/mwaskom/seaborn) - for plotting, if you want to reproduce our figures

## Obtain Datasets

- [AMPDs](http://ampds.org) - water, gas, and power consumption of a home over time
- [MSRC-12](http://research.microsoft.com/en-us/um/cambridge/projects/msrc12/) - Kinect readings as subjects performed various actions
- [PAMAP](http://www.pamap.org/demo.html) Physical activity monitored by on-body sensors
- [UCI Gas](http://archive.ics.uci.edu/ml/datasets/gas+sensor+array+under+dynamic+gas+mixtures) - measurements of gas concentrations over time
- [UCR Archive](http://www.cs.ucr.edu/~eamonn/time_series_data/) - A collection of 85 time series datasets

## Run Experiments

 1. Clone our [benchmark repository](https://github.com/dblalock/lzbench).
 1. Modify `_python/datasets/paths.py` to point to dataset locations on your machine.
 1. Modify `_python/config.py` to save results and figures where you'd like.
 1. Run
    ```
        $ python -m _python.main
    ```
    with the command line flag for the experiment you want to run. Flags are shown at the bottom of `_python/main.py`.
  1. Once the experiment has run, you can uncomment the appropriate figure creation call at the bottom of `_python/figs.py` and run:
    ```
        $ python -m _python.figs
    ```


And yes, it would be better if there were scripts to curl all the datasets into a local directory and command line arguments to create all the figures. Pull requests welcome.

## Look at results directly

As an alternative to re-running the experiments if you just want to compare to us, you can look at our raw numbers in the `results/` directory. Using our benchmark code is highly recommended though since it will probably make profiling your algorithm in comparison to others much easier.

# Additional experimental details

- We removed the timestamps from all datasets for our experiments, since 1) timestamps are often baked into the indexing scheme in time series databases, 2) in practice, one often knows a priori that timestamps are uniformly spaced, and so can just store the start time for a given block of data, 3) there exist specialized schemes for storing timestamps and we don't claim to outperform these schemes, and 4) using timestamps would make the results less interpretable by "diluting" the characteristics of each dataset.
- For the integer compression schemes FastPFOR, SIMDBP128, and Simple8B, we zero-padded the data to 32 bit integers, since this is what these methods require. We obtained the reported compression ratios by dividing the raw ratios by padding factor (i.e., by 4 for 8-bit data and by 2 for 16-bit data). We tried running them on the raw byte streams without padding and, unsurprisingly, they achieved virtually no compression.
- We also tried many other compressors not included in the paper, including Brotli, Blosc with Byteshuffle, Blosc with Bitshuffle, LZ4HC, FSE, LZO, and others. We selected the reported algorithms on the basis that they were (generally) on the Pareto frontier of ratio vs decompression speed and were in common use in time series databases. Also, adding more algorithms clutters the figures and makes it nearly impossible for to obtain statistically meaningful comparisons thanks to multiple hypothesis testing. As an important note for the latter purpose, we decided to use this subset *before* running our final experiments.
- We omitted a few experiments to tighten up the results section. Probably the most interesting of these is that profiling the compression ratios of various methods using different block sizes. Results for 1KB and 10KB blocks are shown below. These illustrate that, in the presence of extremely limited memory, Sprintz does even better relative to other methods.

![Sprintz-1KB](/communicate/assets/boxplot_ucr_1KB.png?raw=true)
![Sprintz-10KB](/communicate/assets/boxplot_ucr_10KB.png?raw=true)

# Notes

- At present, Sprintz has only been tested with Clang on OS X.
- Feel free to contact us with any and all questions. We're happy to help.
