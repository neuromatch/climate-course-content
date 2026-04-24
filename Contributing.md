# Workflow

The Course is built off from the `main` branch of this repo. The `main` branch is protected and only [code owners](https://github.com/orgs/ClimateMatchAcademy/teams/content-code-owners) can approve Pull Requests.

## Prerequisites
1. Have a Github accounts and request @WesleyTheGeolien to add you to [ClimateMatchAcademy](https://github.com/orgs/ClimateMatchAcademy/teams) and [Content Creators](https://github.com/orgs/ClimateMatchAcademy/teams/content-creators)
1. Have git (a version control system) installed locally, see documnetation [here](https://github.com/git-guides/install-git), if you are not familiar with using a terminal I would recommend installing the Github Desktop Client
1. (Optional) Install [VS Code](https://code.visualstudio.com/download) which is your "development environment", it also intergrates nicely with Version Control (git)
1. (Optional) Install [miniforge](https://github.com/conda-forge/miniforge) to run notebooks locally. This is used to create virtual environments and execute your notebooks.

## Running Notebooks Locally

To run tutorial notebooks on your own machine:

1. Install [miniforge](https://github.com/conda-forge/miniforge) if you haven't already
1. Clone the repo and create the conda environment:
   ```bash
   git clone https://github.com/neuromatch/climate-course-content.git
   cd climate-course-content
   conda env create -f environment.yml
   conda activate climatematch
   ```
1. Launch Jupyter and open any tutorial notebook:
   ```bash
   jupyter lab
   ```

The `environment.yml` at the repo root contains all packages needed to run every tutorial.

## Updating Tutorial Notebooks

1. First you will need to create a new branch for your work (this can be done locally if you are familiar with using git or from the [interface](https://github.com/ClimateMatchAcademy/course-content/branches) and clicking new branch), please give it a reasonable name!
1. Next you have the choice of editing the file locally or online, for quick edits online is quicker and easier but you can not run your notebook / test your notebook.
    1. __Locally__: Please follow these [instructions](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) to clone the repo then checkout your branch and start work, (see this [video](https://www.youtube.com/watch?v=H5BLEPhqxe8) for how to change branch inside VS Code, if you need extra help ask @WesleyTheGeolien)
    1. __Online__: On the [Repo homepage](https://github.com/ClimateMatchAcademy/course-content) click the icon top left that says "main" and choose your branch. All the code should have updated. If you look at the url it will have changed as well (for example https://github.com/ClimateMatchAcademy/course-content/tree/pre-pod) we _just_ need to change github.com to github.dev and you will have an online VS Code environemnt to make your changes
1. Carry out your work
1. Create a [Pull request](https://github.com/ClimateMatchAcademy/course-content/compare) for your branch and request review
1. Upon Review your work will be merged into the main branch