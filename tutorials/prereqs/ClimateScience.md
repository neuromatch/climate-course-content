# Prerequisites and preparatory materials for Computational Tools for Climate Science

Welcome to [Climatematch Academy](https://neuromatch.io/climate-science)! We are very excited to bring *Computational Tools for Climate Sceince* to such a wide and varied audience. We want to make sure every student is able to follow and enjoy the materials presented during the course. In order to get the most out of the materials we offer, we expect our students to know the basics of programming in Python, as well core concepts in math and science. Below 👇 we provide more details for each.

## Contents

* [Programming](#Programming)
* [Algebra](#algebra)
* [Linear Algebra](#linear-algebra)
* [Statistics](#statistics)
* [Calculus](#calculus)
* [Physics](#physics)
* [Chemistry](#chemistry)
* [Climate Science](#climate-science)



## Programming


We expect students to be familiar with fundamental Python and data storage concepts (variables, lists, dictionaries, data formats) as well as some key Python libraries (NumPy, matplotlib, cartopy, datetime, pandas, XArray).  If these feel a little unfamiliar, you are coming from another programming language, or you just want to make sure you are up to speed for the course, we **highly recommend** you take our **Python Refresher**. It is a selection of Project Pythia tutorials (outlined below) that you will work through asynchronously *before* the course begins at your own pace.

What is Project Pythia? Check out the fantastic [~5min video](https://bit.ly/42P799Y) that Julia Kent prepared for you as an intro to this fabulous material.

Below are the lessons you need to review. **To get started with a tutorial just click on the rocket icon at the top right of each tutorial notebook and then click the 'Binder' button that appears ro launch the tutorial (see image below and more instructions [here](https://foundations.projectpythia.org/preamble/how-to-use.html#interacting-with-jupyter-notebooks-in-the-cloud-via-binder))!** If you’re from a Matlab background, you may find reviewing [this cheatsheet](https://cheatsheets.quantecon.org/) helpful before you get started.

![Launch Binder](binder.png)

### Preamble and Foundational Skills (~3 hours)

| Tutorial Section and Link | Approximate Time to Complete (minutes) |
| ----------- | ----------- |
| [Interacting with Jupyter Notebooks in the cloud via Binder](https://foundations.projectpythia.org/preamble/how-to-use.html#interacting-with-jupyter-notebooks-in-the-cloud-via-binder)  | 10 |
| [Why Python?](https://foundations.projectpythia.org/foundations/why-python.html) | 15  |
| [Quickstart: Zero to Python](https://foundations.projectpythia.org/foundations/quickstart.html) | 90 |
| [Jupyterlab](https://foundations.projectpythia.org/foundations/jupyterlab.html) | 50 |
    
### Core Scientific Python Packages (~6 hours)
    
| Tutorial Section and Link | Approximate Time to Complete (minutes) |
| ----------- | ----------- |
| [Overview](https://foundations.projectpythia.org/core/overview.html)  | 10 |
| [NumPy](https://foundations.projectpythia.org/core/numpy.html) | 85  |
| [Matplotlib](https://foundations.projectpythia.org/core/matplotlib.html) | 100 |
| [Cartopy](https://foundations.projectpythia.org/core/cartopy.html) | 30 |
| [Datetime](https://foundations.projectpythia.org/core/datetime.html) | 30 |
| [Pandas](https://foundations.projectpythia.org/core/pandas.html) | 60 |
| [Data Formats](https://foundations.projectpythia.org/core/data-formats.html) | 50 |
| Optional (will be covered in the first day of course materials): [Xarray](https://foundations.projectpythia.org/core/xarray.html) | 150 |

To ensure clarity and set proper expectations, let’s revisit what you can expect from the **Python Refresher Materials** and what you should not expect.

### What You Can Expect from the Python Refresher

- Self-Study Approach: The refresher is *not* designed as a traditional course with lectures and guided lessons. Instead, we have curated a collection of comprehensive learning materials for you to study independently.  It is essential that you dedicate sufficient time and effort to self-study the content at your own pace through the links provided above.
    
- Prior Programming Experience Advantage: If you have a large degree of prior experience in programming, you may find this refresher relatively easy to grasp. 
    
- For those with less of a programming background, it is crucial to study the material well in advance. We recommend beginning as early as possible to ensure you have ample time to cover the content. Practice every day and you'll be in great shape before the course begins.
    
- You won't need to install Python on your computer for the Python refresher. All you need to do is click the rocket-shaped button in the upper right corner of the lessons.

- The section [“How to use this book”](https://foundations.projectpythia.org/preamble/how-to-use.html#how-to-use-this-book) provides you with alternative options to run the Python code should you need to.

- If nothing works, please don’t panic. Sometimes depending on your region, your internet access, and your computer, some challenges may arise, but we will be happy to assist you. Please email  [nma@neuromatch.io](mailto:nma@neuromatch.io) if you need help.
    

### What You Should Not Expect from the Python Refresher

- Guided Instruction: Please be aware that the Python refresher does not include virtual online classes or guided instruction.  It is on you to study the material independently and seek clarification as needed from existing online resources.

- Traditional Classroom Environment: Unlike traditional courses where you attend classes and follow a set schedule, the refresher relies on your self-motivation and discipline to study the provided materials. There will not be any fixed class times or mandatory attendance.

- In-depth Coverage of Advanced Topics: The Python refresher primarily focuses on introducing the fundamentals of Python programming. It does not delve into advanced topics. We recommend the [Software carpentry 1-day Python tutorial](https://swcarpentry.github.io/python-novice-inflammation/) or the free Edx course [Using Python for Research](https://www.edx.org/course/using-python-for-research). For a more in-depth intro, see the [scipy lecture notes](https://scipy-lectures.org/). Finally, you can follow the [Python data science handbook](https://jakevdp.github.io/PythonDataScienceHandbook/), which also has a print edition. 

    
## Math

Climatematch Academy relies on linear algebra, probability, basic statistics, and calculus (derivatives, integrals, and ordinary differential equations -ODEs).

### Algebra
If you need a refresher on basic, pre-calculus algebra check the videos in this [Algebra course](https://bit.ly/3Pnq5cP) from Khan Academy. You will particularly need to know your way around [functions](https://bit.ly/43H5mox) (unit 8 of the course) and their representations, including graphics, because they are the way to express relationships between variables and they are the basic elements you work with in Calculus. 

### Linear algebra
You will need a good grasp of the basics of linear algebra to follow along, as linear algebra is crucial for almost anything quantitative involving more than one number at a time. It will also help you visualize and understand the way data are organized and manipulated, particularly in computational environments. You need to have a basic understanding of vectors and matrices, and how to perform operations with them. We recommend watching the videos in these units about [vectors](https://bit.ly/3NcM2IK) and [matrices](https://bit.ly/3CDXurX) from Khan Academy. That should be enough to prepare you for CMA. 

> Extra material. 
>
>If you get curious and want to build a strong background in linear algebra, we recommend [W0D3](https://compneuro.neuromatch.io/tutorials/W0D3_LinearAlgebra/chapter_title.html) from Neuromatch Academy, or this [beautiful lecture series](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab). Another great resource is this [Linear Algebra Course](https://www.khanacademy.org/math/linear-algebra) from Khan Academy. Here is a series of exercises on [linear algebra in Python](https://www.w3resource.com/python-exercises/numpy/linear-algebra/index.php).

  

### Statistics
Understanding statistics is also important; you should be comfortable with mean/median/mode, standard deviations, variances, the normal distribution, and linear regression. We recommend the [Statistics and probability course](https://bit.ly/3CGx6h2) (videos in units 3, 4 and 5) from Khan Academy. 

  

### Calculus
Finally, basic calculus is crucial; you should know what integrals and derivatives are, and understand what a differential equation means. If you need to refresh your memory on differential and integral calculus, we recommend the [Calculus course](https://bit.ly/3Nk4sXV) from Khan Academy (videos in units 2, 5, 6, and 7). Remember that you don’t need to learn all the details and be able to solve complicated problems, just make sure you understand the concepts. We also recommend [these simulations](https://phet.colorado.edu/en/simulations/filter?subjects=math&levels=university&type=html,prototype) from PHET to help illustrate some mathematical concepts.

> Extra material:
>  
 >For additional reading, [Gilbert Strang's book](https://ocw.mit.edu/ans7870/resources/Strang/Edited/Calculus/Calculus.pdf) is a good refreshment book: Chapters 1 (sections 1.1-1.3), 2, 3 (sections 3.1-3.4), 5, 13 (sections 13.1-13.2), and 16 (section 16.2). This book also has chapters on Vectors and Matrices (Chapter 11), and Linear Algebra (section 16.1). For differential equations, we also recommend reading sections 0.2 and 0.3 of Jiri Lebl's book ["Differential equations for engineers"](https://www.jirka.org/diffyqs/). 

  

## Science

  

### Physics
Climate processes are governed by the laws of physics which is why you will need a general understanding of physics concepts such as: Newton’s laws of motion, forms of energy, conservation of energy, circular motion (Earth’s rotation and Coriolis force), waves, electromagnetic spectrum, optics, heat, and thermodynamics, etc.

Recommended references include the videos from the [College Physics 1](https://www.khanacademy.org/science/ap-college-physics-1) course from Khan Academy for kinematics, Newton’s laws, dynamics, energy, and mechanical waves), the videos from the [College Physics 2](https://www.khanacademy.org/science/ap-physics-2) course from Khan Academy for Heat and Thermodynamics (unit 2), electromagnetic waves (unit 6), and optics (unit 7), and [Dave Van Domelen](https://stratus.ssec.wisc.edu/courses/gg101/coriolis/coriolis.html)’s, [Encyclopedia Britannica](https://www.britannica.com/science/Coriolis-force) or [Wikipedia](https://en.wikipedia.org/wiki/Coriolis_force) entries on Coriolis force. 

Remember that you don’t need to learn all the details and be able to solve complicated problems, just make sure you understand the concepts. We also recommend [these simulations](https://phet.colorado.edu/en/simulations/filter?subjects=motion,sound-and-waves,work-energy-and-power,heat-and-thermodynamics,light-and-radiation&levels=university&type=html,prototype) from PHET to help illustrate some concepts.

 > Extra material:
>  
> For a more comprehensive refresher, we recommend Urone, Hinrichs, and Dirks’ [College Physics book](https://open.umn.edu/opentextbooks/textbooks/61): for Newton’s laws of motion (chapter 4 up to section 4.4), circular motion (chapter 6 up to section 6.4), conservation of energy (chapter 7 up to section 7.6), heat and thermodynamics (chapter 13 except section 13.4, chapter 14, chapter 15 except sections 15.2, 15.5, and 15.7), waves (Sections 16.9-16.11), electromagnetic spectrum (sections 24.3, 24.4), optics (chapter 25 up to section 25.5). 

  
  
  

### Chemistry
It is important to have general chemistry knowledge regarding atoms, isotopes, molecules, ions, compounds, bonds, etc. We recommend the [College Chemistry course](https://www.khanacademy.org/science/ap-chemistry-beta) (videos in units 1, 2, and 3) from Khan Academy. There is no need to learn all the details and be able to solve complicated problems, just make sure you understand general concepts. We also recommend [these simulations](https://phet.colorado.edu/en/simulations/filter?subjects=general&levels=university&type=html,prototype) from PHET to help illustrate some concepts.

 > Extra material:
 > 
 >For additional reading, we also recommend Flowers, Theopold, Langley, and Robinson’s [Chemistry 2e](https://openstax.org/details/books/chemistry-2e) book (sections 1.2, 1.3, 2.3-2.6, 2.7 is optional). 

  

### Climate Science

If you're coming from outside climate science, it'll be great to familiarize yourself with fundamental concepts. Watch this [video](https://bit.ly/3NEDdsJ), to build some context for CMA. As a good accompanying resource for CMA and if you’d like to learn more about climate science, we highly recommend Andreas Schmitner’s [Introduction to Climate Science](https://open.umn.edu/opentextbooks/textbooks/860) book.

  
  

We're excited to have you here! Looking forward to meeting you soon,

  

The Climatematch Academy Team.
