# Solid state detector for X-rays (ASIX)

The followind logbook contains informations about the solid state detector to be implemented as an upgrade of the GPD for X-rays.  
Instead of the gas as conversion stadium, a solid state conversion stadium will be inserted. 

### ASIX proposal review

![Scheme of the detector](figures/asix_scheme.png)

- Goal: This proposal aims at establishing new X-ray detection systems with simultaneous optimal imaging and spectral capabilities. Position and energy resolution shall be better than:
- 10 𝝻m
- 300 eV, at a maximum global readout rate of 100kHz and energy range of 1-50 keV. 


The ASIX systems will implement a hybrid structure combining:
- a core front-end, large area CMOS ASIC with 50𝝻m pitch hexagonal pixels featuring
signal triggering and smart processing logic;
- a customized silicon sensor with appropriate geometry for mating with the ASIC pixels;
- a simple, robust electro-mechanical integrated assembly.

![Principal features of the detector](figures/asix_features.png)

The multiplication does not occur as in gas detectors. Signal pixels are few, less than 10. 
The main parameters in a detector like this one are the following:
- width of the SSCS (Solid State Conversion Stadium);
- noise of the readout chip (that is, by now, XPOL-III);
- Fano factor, that limits the energy resolution toghether with noise. 

Having no multiplication of signal, the detector is no more in barely noiseless conditions as in the GPD.  
**Noise sums in quadrature** and is no longer as tiny wrt single pixel signal as in GPD case. (signal is the sum of pixel signals). 
The range of particles is a lot shorter than the gas case. 

This chip will have different applications from IXPE, in particular can be used for crystals and in general material analysis. For this tasks, event rate should increase dramatically (of some orders of magnitudes).

Because of those main issues, the characteristics to be tuned opportunely are:
- keeping noise as low as possible;
- higher chip readout.

First task: resolve K-lines of copper. Those lines are at:
- 8.046 keV (principal K-$\alpha$ line)
- 8.904 keV (principal K-$\beta$ line)

![Cu forest](figures/Cu_K_line_forest.png)

It is desirable to divide the Ka1+Ka2 from the Kb1+Kb3+Kb5. $\Delta E \simeq 880$ eV.  

Note that resolving the lines is not only an hardware tuning, the **reconstruction can be tuned for obtaining peak division**. 

## Event physics 
As previously described, this detector works using photoelectric effect of $\gamma$ on the silicon detector.  
The physics with zero noise is easy to compute:  
Being $E_{K\beta} = 8900$ eV, it is easy to compute the mean number of electron-holes pair created by the photoconversion: $<e> = \frac{E_{K\beta}}{E_{ion}} \pm \sqrt{\frac{E_{K\beta}}{E_{ion}} \cdot F}= 2472 \pm 17$ [$e-$]  
(in Si, where $E_{ion}=3.6$ eV and $F\simeq 0.128$).  

This results in the following energy resolution in eV: $R = <e> \cdot E_{ion} = 60$ eV. 

The principal noise effect is due to _charge sharing_.  
With _charge sharing_ it is intended the process of diffusion of the cloud in the medium that cause the electrons to trigger the bordering pixels to the one (or more) where the cloud was formed.  
This effect damages signal because the electronic noise sums in quadrature:
$$\sigma_{tot} = \sqrt{\sum_{i=1}^{N_{signal}}(\sigma_{<e>} + \sigma_{electronics})^2} $$  

Where:
- $\sigma_{<e>}$ is the uncertainty due to statistical fluctuations ($\sigma_{<e>} = 17$ [$e-$]);
- $\sigma_{electronics}$ depends on chip readout and it is measured in ENC (Equivalen Noise Charge = signal charge in electrons that determines a $\frac{S}{N} = 1$).


## Analysis of MC events - defining the interesting metrics

Main idea is starting from any reconstruction algorithm and searching for hardware tuning (searching for physical parameters of the detector). After hardware tuning, the reconstruction algorithm can be changed in order to find the most efficient. 

The parameters of the simulation regard the silicon detector:
- _thickness_ ($t$) of the silicon detector (ranges from 50 $\mu m$ to 500 $\mu m$);
- _pitch_ ($p$) of the readout custom chip, that defines the size of every hexagonal pixel (ranges from 50 $\mu m$ to 100 $\mu m$).  
Remember that we use hexagonal grids as we can see in the figure below.  
So, if we call pitch $p$ the _vertical spacing_, then the pitch in _horizontal direction_ is $\frac{\sqrt{3}}{2}\cdot p$ (For further reading, see: [hexagon_grid_link](https://www.redblobgames.com/grids/hexagons/));  

![Hexagonal grid](figures/hexagonal_grid.png)

- _noise_ ($N$) of every pixel in the readout custom chip [ENC].

The tradeoff to find regards the parameters $t$ and $p$.  
A low **thickness** $t$ implies a _lower photon efficiency_ but an higher thickness implies an _higher charge sharing_, that means _higher noise_.  
A low **pitch** $p$ implies higher granularity of the readout and so signals of more pixels, that could be useful for reconstruction of direction (for polarimetry) but could be dispersive in terms of energy reconstruction.  

The useful metrics for measuring the goodness of the detector are:
- energy resolution $R = \frac{\Delta E}{E}$ and _effective energy resolution_ (?);
- rejection power ($RP$) for now defined as: $$RP =\frac{\text{number of } \gamma \text{ identified as contamination}}{\text{number of } \gamma \text{ that are contamination}}$$
This quantity is more interesting than the _fraction of contamination_ because it is properly normalized. ASK BALDINI 
- quantum efficiency $QE$:
$$QE = \frac{\text{ number of emitted } e-}{\text{number of incident }\gamma}$$
Note that this is slightly different from the _photoabsorption efficiency_ of a material, that is the probability that a single photon is absorbed via photoelectric effect.
- signal noise ratio $\frac{S}{N}$;

- fraction of events with a single signal pixel wrt ALL EVENTS:
$$f_{1px} =N_{1px} \cdot \frac{\epsilon_{\gamma}}{N_{detected}}$$
Where:  
-> $N_{1px}$ is the number of event detected with 1 signal pixel;  
-> $\epsilon_{\gamma}$ is the photoabsorption associated with the silicon detector (dependent for its thickness);  
-> $N_{detected}$ is the number of detected events.

### Interesting plots by now
- selecting events by number of signal pixels, looking at how the mean and the sigma of the distribution of peaks change;
- plot of noise vs thick vs energy resolution




## What to do next?
- Redo computations of Carmelo and Luca: [x]
![Measures to redo](figures/measures.png) 

    Measures done for both $K_{\alpha}$ and $K_{\beta}$. Plots to be done:
    - Colormap thick-noise-relative variation wrt true energy value [x] (understand how to plot more colormaps together);
    - Colormap thick-noise-$\frac{FWHM}{\mu}$ [x].
    - Add numbers in the heatmaps [x]


## Hexsample simulation grid 
The parameter space taken into consideration is _thickness-ENC-pitch_.

Where:
- _thickness_ is a feature of the solid state detector;
- _ENC_ and _pitch_ are features of the readout custom chip.

In order to represent 2D plots, the 3 possible couples of parameters has been exploited, fixing the third parameter to a reasonable value (often conservative ones).

### thickness-pitch space 
In the following grid of parameters, ENC value has been set. $\text{ENC} = 40$ enc.  

#### Fraction of events with a single px

![f](figures/Figure_5.png)

The trend is the one expected:
- as the __pitch increases__, the number of events collected completely in a single pixel __increases__;
- as the __thickness increases__, the diffusion effect is bigger, so there are less events collected completely in a single pixel.

#### Energy resolution in events with a single pixel
Energy resolution in number of electrons is shown in the following figure:  

![f](figures/Figure_4.png)

Energy resolution is around 2-3%, that corresponds to 178-267 eV for $\alpha$ peak and 160-241 eV for $\beta$ peak. It is barely constant with pitch (it is slightly better for bigger pitch but of a negligible quantity, few electrons. As written above, higher pitch means higher probability of containing the track completely. If pitch is smaller, there are more border zones where part of the signal can be lost.)

The values are lower than the ones shown in the reference measures done by Luca and Carmelo. _need to understand why_

#### Mean shift in events with a single pixel
![f](figures/Figure_3.png)
For every value, it is negliglible for both $\alpha$ and $\beta$ (anyway always negative, that makes sense).

#### Energy resolution for all events
Energy resolution in number of electrons is shown in the following figure:  
![f](figures/Figure_2.png)  

The values are lower than the ones shown in the reference measures done by Luca and Carmelo. _need to understand why_  

As for the 1 px case, energy resolution is not strongly dependent to pitch, same for mean shift, shown below.


#### Mean shift in events with a single pixel
![f](figures/Figure_1.png)

Some other metrics of interests are:
- quantum efficiency
- think about it. 

