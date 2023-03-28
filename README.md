# ml_fnl_forecast
Public code for the forecasts computed in [arXiv:2303.08901](https://arxiv.org/abs/2303.08901)

## To use:

<ol>
<li>Install <a href="https://git-scm.com">Git</a> and <a href="https://www.python.org">python 3</a>.</li>
<li>Clone this repository </li>

    $ git clone git@github.com:jmsull/ml_fnl_forecast.git
<li>Make a virtual environment (optional, recommended). For this step install <a href="https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html">conda</a></li>

    $ conda create -n envname python=x.x anaconda
    $ conda activate envname
    
<li>Download dependencies</li>

    $ conda install --yes --file requirements.txt

<li>Download snapshot from <a href="https://www.tng-project.org/data/downloads/TNG300-1/">IllustrisTNG</a> and copy it inside <a href="b_phi_predictions/data/TNG-300/">TNG-300/</a></li>
<li>Download <a href="https://www.tng-project.org/api/TNG300-1/files/halo_structure/">supplementary halo structure catalog</a> and copy it inside <a href="b_phi_predictions/data/TNG-300/supplementary_catalog/">supplementary_catalog/</a></li>

<li>To get galaxy samples</li>
    
    $ python3 b_phi_predictions/sample_selection/main.py
</ol>


