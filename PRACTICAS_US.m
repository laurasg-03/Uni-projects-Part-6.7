%% PRÁCTICAS ULTRASONIDOS

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. CREAR LA REJILLA K-SPACE
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% computational grid, the simulations are performed on a regular Cartesian mesh
clc;
close all;
clear all;
% Total number of grid points (Nx, Ny) 
% Spacing between the grid points (dx, dy) 
% !!! Based on FFT --> faster when N is given by a power of two or has small prime factors. 
% X REFER TO THE POSITION OF A ROW AND Y TO THE POSITION OF A COLUMN.

% create the computational grid
Nx = 128;           % number of grid points in the x (row) direction
Ny = 128;           % number of grid points in the y (column) direction
dx = 0.1e-3;        % grid point spacing in the x direction [m]
dy = 0.1e-3;        % grid point spacing in the y direction [m]
kgrid = kWaveGrid(Nx, dx, Ny, dy);
% definimos una cuadrícula en la que se simula la propagación de ondas
% devuelve una estructura de datos que contiene información sobre la cuadrícula, como las coordenadas de los puntos de cuadrícula y la resolución espacial.

kgrid.Nx; % El número de puntos de cuadrícula en la dirección x.
kgrid.dx; % El espaciado entre los puntos de cuadrícula en la dirección x.
kgrid.Ny; % El número de puntos de cuadrícula en la dirección y.
kgrid.dy; % El espaciado entre los puntos de cuadrícula en la dirección y.
kgrid.x; % Las coordenadas de los puntos de cuadrícula en la dirección x.
kgrid.y; % Las coordenadas de los puntos de cuadrícula en la dirección y.

kgrid.kx_max; % La frecuencia máxima de onda en el dominio de Fourier en la dirección x.
kgrid.dt; % El paso de tiempo utilizado en la simulación.
kgrid.x_size; % El tamaño total del dominio de la simulación en la dirección x.
kgrid.y_size; % El tamaño total del dominio de la simulación en la dirección y.
kgrid.total_grid_points; % El número total de puntos de cuadrícula en la cuadrícula.

%% 2. Defining the acoustic parameters (speed of sound, medium density, and acoustic absorption).
% For a homogeneous medium, the sound speed is set as a scalar value in m/s). 
% Power law acoustic absorption --> set by assigning values to medium.alpha_coeff (dB/(MHz^y cm)) and medium.alpha_power (power law exponent)
% The density of the medium is defined in medium.density. 
% For heterogeneus media homogeneous media all these factors can be set as a matrix which contains the parameter values for each grid point (same size as the medium discretisation defined by the computational grid (Nx y Ny).

% define the properties of the propagation medium
medium.sound_speed = 1500;  % [m/s]
medium.alpha_coeff = 0.75;  % [dB/(MHz^y cm)]
medium.alpha_power = 1.5;   % factor to adjust the power law for limiting frequency ranges. Es la absorcion
medium.density=1000;        % [kg/m^3]

f_max_x = medium.sound_speed/(2*dx); %Frecuencia máxima que la malla puede propagar (depende del espaciado de la malla, dx y dy, y de la velocidad del sonido)

%% 3. Defining the distribution of the initial pressure
% Initial pressure distribution --> matrix (Nx, Ny) which contains the initial pressure values for each grid point. 
% In this example, the function makeDisc is used to create an initial pressure distribution in the shape of a small discs.
% This distribution is assigned to the p0 field of the source structure (source.p0 except must be Nx, Ny).

% create initial pressure distribution using makeDisc
disc_magnitude = 5; % [Pa]
disc_x_pos = 20;    % [grid points] Se mueve en el eje de las y
disc_y_pos = 40;    % [grid points] Se mueve en el eje de las x
disc_radius = 3;    % [grid points] Cómo de grande es el disco
disc_1 = disc_magnitude * makeDisc(Nx, Ny, disc_x_pos, disc_y_pos, disc_radius);

source.p0 = disc_1; % Presión inicial

figure; imagesc(source.p0);
colormap("gray")

%% 4. Defining the distribution of the sensor/s
% The sensor mask defines the positions within the computational domain where the pressure field is recorded at each time-step. Can be given in one of three ways:
%  1. A binary matrix which specifies the grid points that record the data (same size of the computational grid).
%  2. The grid coordinates of two opposing corners of a line (1D), rectangle (2D) or cuboid (3D).
%  3. A set of Cartesian coordinates lying within the dimensions of the computational domain.
% define a point sensor at the center of the grid placed at the mid lower part.

sensor.mask=zeros(Nx,Ny); %  1. Se inicializa el sensor en cero (no hay sensores activados)
sensor.mask(Nx-Nx/4,Ny/2)=1; %  2. Se activa el único punto del sensor que se activa (en Nx-Nx/4,Ny/2)
% El sensor está diseñado para registrar el campo de presión en cada paso de tiempo en ciertas ubicaciones dentro del dominio.

%% 5. Running the simulation.
clc
% kspaceFirstOrder2D function runs the simulation automatically, giving as ouput the pressure signals collected at each sensor point
% Pressure at the Cartesian points in 2D is computed at each time step using linear interpolation and the temporal evolution of the pressure wave can be observed. 

% arguments for kspaceFirstOrder2D in order to record a movie. We choose not to show live the simlulation in mlx (looks bad).
input_args = {'RecordMovie', true, 'MovieName', 'simulacion 1','MovieArgs',{'FrameRate', 10},'PlotSim', false,'PlotLayout', false};  
% run the simulation 
sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor,input_args{:});  % devuelve sensores,datos de presion y tiempo
% Durante la ejecución, se imprimen actualizaciones de estado y parámetros computacionales en la línea de comando. Esto puede incluir información sobre el progreso de la simulación, los parámetros utilizados y cualquier otro mensaje relevante para el usuario.
% Después de completar el bucle de tiempo de la simulación, la función devuelve los datos registrados por el sensor en cada punto definido por sensor.mask. Estos datos se devuelven en forma de series temporales, donde cada fila corresponde a un punto del sensor y cada columna corresponde a un paso de tiempo (sensor_data(sensor_point_index, time_index)).
implay('simulacion.avi');

%% plot the simulated sensor data
figure;
plot(kgrid.t_array*1e6, sensor_data(1,:)); xlabel('Time (\mus)'); ylabel('Pressure (Pa)');

%% Q1. Plot the source distribution, the detector distribution and display the first five terms of the x positions of the grid
% Grafica la Distribución de la Fuente
figure;
subplot(1,2,1); 
imagesc(kgrid.x_vec*1e3, kgrid.y_vec*1e3, source.p0); xlabel('y (mm)'); ylabel('x (mm)'); title('source');
axis('square');
colormap('gray');

% Grafica la Distribución del Sensor
subplot(1,2,2); 
imagesc(kgrid.x_vec*1e3, kgrid.y_vec*1e3, sensor.mask);xlabel('y (mm)'); ylabel('x (mm)'); title('detector');
axis('square');
colormap('gray');

display(kgrid.x_vec(1:5));

%% Q2. Run the simulation again but this time set the speed of sound of the medium 
clc;
medium.sound_speed =1500.*ones(Nx,Ny);                        % define the speed sound as 1500 m/s in the whole grid
medium.sound_speed(Nx/2:end,:)=2000;                          % define the speed sound as 2000 m/s in the whole grid
input_args = {'RecordMovie', true, 'MovieName', 'simulacion 2','MovieArgs',{'FrameRate', 10},'PlotSim', false, 'PlotLayout', false};
sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor,input_args{:});
% Se observa reflexión de las ondas en la interfaz entre las dos regiones. Parte de la energía de la onda puede reflejarse de vuelta a la región de menor velocidad del sonido (1500 m/s), mientras que otra parte puede transmitirse a la región de mayor velocidad del sonido (2000 m/s).


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2. CREAR LA REJILLA K-SPACE
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
clear all;
clc; 

% Creación de la Cuadrícula Computacional
Nx = 128;           % number of grid points in the x (row) direction
Ny = 128;           % number of grid points in the y (column) direction
dx = 0.5e-3;        % grid point spacing in the x direction [m]
dy = 0.5e-3;        % grid point spacing in the y direction [m]
kgrid = kWaveGrid(Nx, dx, Ny, dy);

% Definición de las Propiedades del Medio
medium.sound_speed = 1500;  % [m/s]
medium.alpha_coeff = 0.75;  % [dB/(MHz^y cm)]
medium.alpha_power = 1.5;   % factor to adjust the power law for limiting frequency ranges.
medium.density=1000;           % [kg/m^3]

% Creación de una Distribución de Velocidad del Sonido Modificada
disc_magnitude = 1500; % [m/s]
disc_x_pos = 35;    % [grid points]
disc_y_pos = 64;    % [grid points]
disc_radius = 20;    % [grid points]

% Actualización de la Velocidad del Sonido en el Medio
disc_1 = disc_magnitude * makeDisc(Nx, Ny, disc_x_pos, disc_y_pos, disc_radius);
figure; imagesc(disc_1);

medium.sound_speed =1500.*ones(Nx,Ny);  
medium.sound_speed= medium.sound_speed + disc_1;

% Definición del Sensor
sensor.mask=zeros(Nx,Ny);                                            % refresh sensor mask
sensor.mask(Nx-Nx/4,Ny/2)=1;   

% Definición de la Fuente
source.p0 = 10*sensor.mask;

% Ejecución de la Simulación
sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor);

% Visualización de los Datos del Sensor
figure; plot(kgrid.t_array(300:end),sensor_data(300:end));  


% • ¿Cuál es el diamtetro del circulo? (kwave)
% El diámetro del circulo se calcula usando la siguiente simple formula: disc_radius*2*dx=20cm

% • ¿Cuál es la distancia entre el emisor/detector y el circulo? (kwave)
% La distancia entre el emisor/detector y el centro círculo se calcula: ((Nx-Nx/4)-disc_x_pos)*dx=30.5cm
% La distancia entre el emisor/detector y el superficie del círculo es por tanto 20.5 cm

% • Con datos de presión calcular la distancia del emisor a la superficie del circulo
% Se resuelva calculando d=v*tp/2 donde v es la velocidad del sonido del medio entre el
% emisor y la superficie del círculo. tp es la posición en el tiempo del primer pico positivo.
% d=0.5*1500*2.75*10^-5 = 20.5cm

% • Con datos de presión calcular el diámetro del círculo
% Se resuelve calculando d=vf*(tp’-tp)/2 donde vf es la velocidad del sonido dentro del
% círculo: d= (4.095-2.75)*0.5*3000*10^-5=20.2 cm 


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3. A FOCUSED TRANSDUCER
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. Simulation of the sensitivity field (transmission Field) of a focused transducer. 
% We will design the focused transducer as the section of a circle (sphere in the 3D case), 
% having a radius of 2.5 cm and an aperture of  25 degrees.
clear all;
close all;
clc;

% create the computational grid
Nx = 256;           % number of grid points in the x (row) direction
Ny = 256;           % number of grid points in the y (column) direction
dx = 0.25e-3;       % grid point spacing in the x direction [m]
dy = 0.25e-3;       % grid point spacing in the y direction [m]
kgrid = kWaveGrid(Nx, dx, Ny, dy);

% defining the time parameters of the simulation
kgrid.dt=1.0000e-08;      %time step on the simulation
tend=(2*(0.03+0.01)/1500);    %last time point on the simulation
kgrid.t_array=[0:kgrid.dt:tend];

%defining the acoustic properties of the medium
medium.alpha_coeff = 0.75;  % [dB/(MHz^y cm)]
medium.alpha_power = 1.5;   % factor to adjust the power law for limiting frequency ranges.
medium.density=1000;           % [kg/m^3]
speed_of_sound = 1500;  % [m/s]
medium.sound_speed =speed_of_sound.*ones(Nx,Ny); % [m/s];

%placing the transducer
source_radius = 100;   % Radio del transductor [puntos de la cuadrícula]
arc_angle = deg2rad(180);     % Ángulo del sector (media luna)
source.p_mask = makeCircle(Nx, Ny, Nx/2, Ny/2, source_radius, arc_angle); % Crear la forma del transductor
source.p_mask((Nx/2)-90:(Nx/2)+2,:) = 0;  % Se crea una apertura más amplia, eliminando parte del círculo del transductor en un rango específico en el eje x, lo que permite que las ondas acústicas se emitan en una dirección más específic

figure; imagesc(kgrid.x_vec*1e2, kgrid.y_vec*1e2, source.p_mask); xlabel('cm'); ylabel('cm'); colormap('gray');

%% Q1. Define the detectors
sensor.mask=ones(Nx,Ny); 
% Esto define los detectores en la cuadrícula. Aquí, se establece un detector en cada punto de la cuadrícula, lo que significa que se registrará la presión acústica en cada posición durante la simulación

%% Q2. Display in an image the sensitivity field
%defining the acoustic pulses 
sampling_freq = 1 / kgrid.dt;     % Frecuencia de muestreo [Hz]
tone_burst_freq = 1e6;            % Frecuencia del pulso [Hz]
tone_burst_cycles = 8;            % Ciclos del pulso
source.p = toneBurst(sampling_freq, tone_burst_freq, tone_burst_cycles); % Generación del pulso acústico
% Pulso acústico con una frecuencia de 1 MHz y 8 ciclos de duración.


%Running the simulation
input_args = {'RecordMovie', true, 'MovieName', 'simulation_focused_transducer','MovieArgs',{'FrameRate', 10},'PlotSim', false,'PlotLayout', false,'DisplayMask','off'};
sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});

%Obtaining the sensitivity field
sensitivity_field = max(sensor_data, [], 2);   % Obtener el campo de sensibilidad
sensitivity_field = reshape(sensitivity_field, Nx, Ny); % Remodelar el campo de sensibilidad a las dimensiones de la cuadrícula
% Se calcula el campo de sensibilidad tomando el valor máximo de los datos registrados por los detectores a lo largo del tiempo. Luego, se remodela para tener las mismas dimensiones que la cuadrícula.

figure; imagesc(kgrid.x_vec*1e2, kgrid.y_vec*1e2,sensitivity_field); xlabel('cm'); ylabel('cm');


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4. BEAM STEERING
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation of beam steering from a linear array. The array will have 40 elements and a total length of 1 cm. We assume each element to be "point-like".
% create the computational grid
close all;
clear all;
clc; 
Nx = 256;           % number of grid points in the x (row) direction
Ny = 256;           % number of grid points in the y (column) direction
dx = 0.25e-3;        % grid point spacing in the x direction [m]
dy = 0.25e-3;        % grid point spacing in the y direction [m]
kgrid = kWaveGrid(Nx, dx, Ny, dy);
kgrid.dt=5.0000e-08;
kgrid.Nt=1207;

medium.sound_speed = 1500;  % [m/s]
medium.alpha_coeff = 0.75;  % [dB/(MHz^y cm)]
medium.alpha_power = 1.5;   % factor to adjust the power law for limiting frequency ranges.
medium.density=1000;           % [kg/m^3]


%% Q1. Define and display the linear array
% Definición del Array Lineal:
num_elements = 40;       % Número de elementos en el array
x_offset = 25;           % Posición del transductor en "altura"
source.p_mask = zeros(Nx, Ny);  % Inicializa la máscara del transductor
start_index = Ny/2 - round(num_elements/2) + 1;  % Índice inicial del array
source.p_mask(x_offset, start_index:start_index + num_elements - 1) = 1;  % Define el array lineal en la máscara del transductor

figure; imagesc(kgrid.x_vec*1e2, kgrid.y_vec*1e2, source.p_mask); xlabel('cm'); ylabel('cm'); colormap('gray');


% define the properties of the tone burst used to drive the transducer
sampling_freq = 1/kgrid.dt;     % Frecuencia de muestreo [Hz]
steering_angle = 30;            % Ángulo de dirección [grados]
element_spacing = dx;           % Espaciado entre elementos [m]
tone_burst_freq = 1e6;          % Frecuencia del pulso [Hz]
tone_burst_cycles = 8;          % Ciclos del pulso

% Cálculo de los Desplazamientos del Pulso para cada Elemento del Transductor:
element_index = -(num_elements - 1)/2:(num_elements - 1)/2;  % Índices relativos a los elementos del array

% use geometric beam forming to calculate the tone burst offsets for each
% transducer element based on the element index
tone_burst_offset =  40 + element_spacing * element_index * sin(steering_angle * pi/180) / (medium.sound_speed * kgrid.dt);  % Cálculo de los desplazamientos del pulso para cada elemento

% create the tone burst signals
source.p = toneBurst(sampling_freq, tone_burst_freq, tone_burst_cycles, 'SignalOffset', tone_burst_offset);  % Creación de las señales de pulso

%% Q2. Define the sensor positions.
sensor.mask=ones(Nx,Ny);  

%Running the simulation
input_args = {'RecordMovie', true, 'MovieName', 'beam_steering_simulation','MovieArgs',{'FrameRate', 10},'PlotSim', false,'PlotLayout', false,'DisplayMask','off'};
sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});

%% Q3. Display the sensitivity field  and  the linear array
sensitivity_field1=max(sensor_data,[],2);
sensitivity_field2=reshape(sensitivity_field1,Nx,Ny);
figure; 
subplot(2,1,1);
imagesc(kgrid.x_vec*1e2, kgrid.y_vec*1e2,source.p_mask); xlabel('cm'); ylabel('cm');
axis('square');
subplot(2,1,2);
imagesc(kgrid.x_vec*1e2, kgrid.y_vec*1e2,sensitivity_field2); xlabel('cm'); ylabel('cm');
axis('square');

%% Q4. Display the acoustic pulses  and interpretate them. Hint: Use the function stacked plot
figure;
t_axis = 1e6 *(1:size(source.p, 2)) / sampling_freq ; %microseconds
stackedPlot(t_axis, source.p);
xlabel('Time [\mus]');
ylabel('Signal Number');