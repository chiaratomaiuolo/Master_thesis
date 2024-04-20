# Solid state detector for X-rays (ASIX)

The followind logbook contains informations about the solid state detector to be implemented as an upgrade of the GPD for X-rays.  
Instead of the gas as conversion stadium, a solid state conversion stadium will be used in order to achieve an high resolution detector for _imaging_ and _spectroscopy_ (no more polarimetry, tracks are too short). 

### ASIX proposal review

![Scheme of the detector](figures/asix_scheme.png)

- Goal: This proposal aims at establishing new X-ray detection systems with simultaneous optimal imaging and spectral capabilities. Position and energy resolution shall be better than:
- 10 $\mu \text{m}$ 
- 300 eV, at a maximum global readout rate of 100kHz and energy range of 1-50 keV.


The ASIX systems will implement a hybrid structure combining:
- a core front-end, large area CMOS ASIC with 50𝝻m pitch hexagonal pixels featuring
signal triggering and smart processing logic;
- a customized silicon sensor with appropriate geometry for mating with the ASIC pixels;
- a simple, robust electro-mechanical integrated assembly.

![Principal features of the detector](figures/asix_features.png)

The multiplication does not occur as in gas detectors. Signal pixels are few, less than 10. 
The main _hardware parameters_ in a detector like this one are the following:
- **width of the SSCS** (Solid State Conversion Stadium);
- **noise of the readout chip** (that is, by now, XPOL-III);
- **Fano factor**, that limits the energy resolution toghether with noise (intrinsic statistical fluctuations in the number of electrons, in this sense this is the liminf for the resolution). 

Having no multiplication of signal, the detector is no more in barely noiseless conditions as in the GPD. **Pixel noise sums in quadrature** and is no longer as tiny wrt single pixel signal as in GPD case (signal is the sum of pixel signals). 
The range of particles is a lot shorter than the gas case. 

This chip will be used for crystals and in general material analysis. For this tasks, event rate should increase dramatically (at least a factor 10--100).

The characteristics to be tuned opportunely are:
- pixel noise as low as possible;
- lowen the dead time for enhancing chip readout rate.

**First task: resolve K-lines of copper**. Those lines are at:
- 8.046 keV (principal K-$\alpha$ line)
- 8.904 keV (principal K-$\beta$ line)

![Cu forest](figures/Cu_K_line_forest.png)

It is desirable to divide the Ka1+Ka2 from the Kb1+Kb3+Kb5. $\Delta \text{E} \simeq 880$ eV.  

Note that resolving the lines is not only an hardware tuning, the **reconstruction can be tuned for obtaining peak division**. 

#### Event physics
As previously described, this detector works using photoelectric effect of $\gamma$ on the silicon detector.  
The physics with zero noise is easy to compute:  
Being $E_{K\beta} = 8900$ eV, the mean number of electron-holes pair created by the photoconversion in Si (where $E_{ion}=3.6$ eV and $F\simeq 0.128$) is: $<e> = \frac{E_{K\beta}}{E_{ion}} \pm \sqrt{\frac{E_{K\beta}}{E_{ion}} \cdot F}= 2472 \pm 17$ [$e-$].  

This results in the following liminf for the energy resolution in eV: $R = <e> \cdot E_{ion} = 60$ eV. 

#### Charge sharing
The principal noise effect is due to __charge sharing__.  
With _charge sharing_ it is intended the process of diffusion of the cloud in the medium that cause the electrons to trigger the bordering pixels to the one (or more) where the cloud was formed.  
This effect damages thre signal because the electronic noise sums in quadrature:
$$\sigma_{tot} = \sqrt{\sum_{i=1}^{N_{signal}}(\sigma_{<e>} + \sigma_{electronics})^2} $$  

Where:
- $\sigma_{<e>}$ is the uncertainty due to statistical fluctuations ($\sigma_{<e>} = 17$ [$e-$]);
- $\sigma_{electronics}$ depends on chip readout and it is measured in ENC (Equivalen Noise Charge = signal charge in electrons that determines a $\frac{S}{N} = 1$).

So clearly the best case scenario in terms of energy resolution are _1 px events_, note that anyway that's not the best case scenario for imaging, where events with few pixels permits the usage of estimator statistics such as the barycenter of charge. 


### Analysis of MC events - defining the interesting metrics

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



### Hexsample simulation grid 
The parameter space taken into consideration is _thickness-ENC-pitch_.

Where:
- _thickness_ is a feature of the solid state detector;
- _ENC_ and _pitch_ are features of the readout custom chip.

In order to represent 2D plots, the 3 possible couples of parameters has been exploited, fixing the third parameter to a reasonable value (often conservative ones).


#### thickness-ENC space 
First goal is to understand for which parameters it is possible to divide the $\alpha$ and $\beta$ peaks, then, we want a resolution that can divide 880 eV. 

Because of that, the first quantity to look at is the energy resolution (in eV).  
Fixing pitch at 50 $\mu \text{m}$.

As measure of the energy resolution, we consider the FWHM of the fitted Gaussian peak for the energy, that is $\text{FWHM} = \sigma_E \cdot 2.35$ .

In the following two figures, it is shown the FWHM for all events and events with a single pixel. 

![fwhm_all](figures/fwhm_allevts.png)
As expected:
- __proportional to thickness__ ;
- __proportional to ENC__.  
- It is higher for $\alpha$ peak but it is fine bc those are absolute resolutions, $K_{\alpha}$ has a larger peak.

It is necesary to set a metric and a limit for the _contamination_ of $K_{\alpha}$ on $K_{\beta}$ (or vice-versa but I think this is more incisive).

![fwhm_1px](figures/fwhm_1px.png)

It improves for 1px tracks as expected. 

#### $\Delta = \frac{\mu_E - E_k}{E_k}$, shift of the mean from true value 

In the following heatmaps, the trend for the shift of the mean from its true value is shown. In the first picture, the one for the 1px events, in the latter the one for all events: 

![shift_1px](figures/mean_shift_1px.png)

This figure shows the effect of the zero suppression on tracks, as a matter of fact, the zero suppression threshold is proportional to noise value: `zero_threshold = noise*SIGMA_THRESHOLD`, where `SIGMA_THRESHOLD` is set to 2.  
This means that a part of the tracks of 1px have not completely collected the charge in a single px, instead, have lost part of the charge in another px that has been successively zero suppressed; clearly this effect is bigger for high thickness because in that case diffusion is higher and so there are few 'real' 1px tracks.  
This could give a measure of how many tracks are effectively of 1px (assuming that this is the only contribution to mean shift, that seems reasonable and assuming that the charge lost is `zero_threshold * 0.5`, just mediating dumbly the value).

At 30 ENC (that seems the limit for our electronics), we have:  
`zero_threshold` = 60 $e-$ = 60*3.6 eV = 216.0 eV .  
$\frac{216}{E_{K_\alpha}} = 27 \%$ , $\frac{216}{E_{K_\beta}} = 24 \%$

So, a rough estimation of the 'false' 1px events could be the following:  
- $\alpha$: $f = \frac{0.024}{0.027} = 88 \%$ for $t = 500$ $\mu\text{m}$, where 0.024 come from simulations;
- $\beta$: $f = \frac{0.018}{0.024} = 75 \%$ for $t = 500$ $\mu\text{m}$, where 0.018 come from simulations.  

Those clearly are overestimates because we are assuming that all tracks lose the maximum that they can. 

The hypotesis that the shift of the mean can be explained by this phenomenon is confirmed by the trend of the mean cluster size shown below:
![mean_clu_size](figures/mean_cluster_size.png)

As we can see, the mean cluster size shrinks as noise and thickness grows, this is counterintuitive in terms of the physics but makes sense when we consider the zero suppression effect. 

As another proof, looking at the shift of the mean for all events, we can see that this effect is suppressed: 
![mean_shift_all](figures/mean_shift.png)
That means that the effect of the tracks cut by zero suppression is mediated and approaches zero.  
This makes sense because the events with a single pixel are << number of total events, as we can see below:

#### Fraction of events of 1px
![f_evts_1px](figures/f_1px_evtspng.png)


#### thickness-pitch space 
In the following grid of parameters, ENC value has been set. $\text{ENC} = 40$ enc.  

### How to evaluate the best values
The first key problem is the evaluation of the reciprocal contamination of the two curves. In order to obtain the best efficiency on signal (both $\alpha$ and $\beta$) with the lowest contamination between the two curves.  
The standard possibilities are three:
- Fixing the _contamination_ and look at the correspondent _efficiency_ value;
- Fixing the _efficiency_ and look at the correspondent _contamination_ value;
- Minimizing the _Gini index_. 

Those three does not seem the right choice for our scopes, for different reasons:
- Fixing a threshold could guarantee some standards but it is not possible to find the _best_, because there is no metric to minimize or maximize;
- The Gini index minimization is not what we are searching for, it looks for the minimum impurity inside each group but when the two groups are too close, it results in merging the $\beta$ elements into $\alpha$ peak, because there is no the concept of misclassification inside it, only the idea of impurity inside groups, no matter of which type. 

Anyway, the scope is to distinguish between the two peaks, so, setting a threshold contamination is the right choice. In particular, we want to guarantee a precision (true positive rate) _on alpha peak_, so we fix the contamination of betas on alpha to a low value, such as 2%.


- plot contributo di fano + contributo elettronico [] 


### Simulation of the event readout
It is necessary now to distinguish the XPOL-III family from the on-development XPOL-IV in terms of hardware features: XPOL-IV would implement an high-level hardware parallelization that would reduce the dead time on readout of the apparatus. Indeed, this would mean changing also the readout mechanism, that seems the right choice considering the different sizes of the events inside a GPD (up to 800 pixels) and inside ASIX (less than 10 pixels). 

#### XPOL-IV hardware design broad-terms 
The interesting main point for the readout scheme are the following:
- Pixel division in 8 (16) independent clusters,
- Every independent cluster maps the adjacent pixels of everyone to different ADC channels in order to permit a further parallelization on readout. The minimum number of ADCs required for guaranteeing this feature is 7. 

#### Readout strategy for XPOL-IV

The latter characteristic aforementioned suggests as parallel readout strategy a 7-pixel digitized track: the higher signal pixel above threshold is taken as _center_ and its 6 neigbors are collected, inspite their content (that could also be 0, this would be informative for the position reconstruction).

#### Readout chain in terms of `Python` code
The class that implement the simulation and the classes necessary for the readout, digitalization and I/O writing/reading are in the following repository: https://github.com/lucabaldini/hexsample.
The readout chain for a single event is the following:

- A photon is created using `PhotonList` class, that contains the informations about the source;
- The photon is propagated inside the active medium;
- The event is passed to a `HexagonaReadout*` (it is at this stage that the type of readout has to be defined, different readout are implemented in different `HexagonalReaodut` classes), that contains the trigger facilities, this class returns a `DigiEvent*` corresponding to a certain readout type if the event triggered the ASIC, else returns `None`;
- The `DigiEvent*` contains all the information for the reconstruction of the event and so represents the informations that the chip will write in file;
- The digitized event is passed to a `DigiOutputFile*` that simulates the writing of the binary informations from the chip to some desired format of the output file that will be used for the reconstruction;
- The event is reconstructed and so a `ReconEvent` object is created. 

#### Types of readout
At this time, the implemented readouts are of three different kinds:
- `HexagonalReadoutSparse`, this is the sparsest one, every pixel over trigger threshold is saved with its logical position and number of electrons without any proximity rule (this is kind of readout is useful for Massimo for simulating the chip behaviour). The corresponding `DigiEventSparse` contains a lenght-variable pha array and two corresponding arrays containing (in the right order) the column and row numbers of the corresponding logical coordinates;
- `HexagonalReadoutRectangular`, this is the XPOL-I readout method that implements the ROI. The corresponding `DigiEventRectangular` contains a lenght-variable pha array and a`RegionOfInterest` object that defines the ROI of the event;
- `HexagonalReadoutCircular`, this is the readout that takes advantage of the ADC parallelization: we identify the highest pixel above trigger threshold and then we take its 6 adjacent neighbors. The correspoding `DigiEventCircular` contains a 7-element array and two `int`s, that correspond to the (`col`, `row`) logical coordinates of the highest pha pixel. 


