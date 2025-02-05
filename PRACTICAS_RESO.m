%% PRÁCTICAS RM

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. TRANSFORMADA DISCRETA DE FOUIER
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Q1: Calculate the DFT using the FFT algorithm of the first 8 points of a sine function of period 1 second and  sampled with a frecuency of 8 Hertzs.
% Display the obtained complex numbers. Plot the frequency amplitud spetrum. Plot the double side amplitude spectrum (Hint, use the function fftshift ). 
%clear the workspace in each new execution

clear all;
close all;


Fs=8;                       % Hz, frecuencia de muestreo (par)
N=8;                        % número de muestras
dt=1/Fs; % 0.1250           % s, resolución en tiempo (cada cuanto cogemos muestra? cada 1/8 segundos)  
df=Fs/N; % 1                % Hz, resolución frecuencia (espaciado entre muestras en el dominio de frecuencia), que es 8/8 = 1Hz

t=[0:dt:(N-1)*dt]; % [0,0.125,0.25,0.375,0.5,0.625,0.75,0.875]         % s, time points
f=[-Fs/2:df:Fs/2-df]; % [-4,-3,-2,-1,0,1,2,3]                          % Hz,frequency points
% f=4Hz es la freq. maxima que podemos obtener sin aliasing

x=sin(2*pi*t); % [0,0.707106781186548,1,0.707106781186548,1.224646799147353e-16,-0.707106781186548,-1,-0.707106781186548]             % AU, señal

%calcuate fft 
X=fft(x);  % [1.144237745221967e-17 + 0.000000000000000e+00i,-0.000000000000000 - 4.000000000000000i,1.224646799147353e-16 - 1.110223024625157e-16i,1.915538118220197e-16 - 4.440892098500626e-16i,2.334869823772510e-16 + 0.000000000000000e+00i,1.915538118220197e-16 + 4.440892098500626e-16i,1.224646799147353e-16 + 1.110223024625157e-16i,-0.000000000000000 + 4.000000000000000i]                 %signal in the frecuency domain
display(X);

X(1)=0; X(3)=0; X(4)=0; X(5)=0; X(6)=0; X(7)=0; %corrección error numerico; X(2)=-4j y X(8)=4j
p= abs(X);              % 0     4     0     0     0     0     0     4   %espectro de amplitudes, tal cual sale de la fft

                                        
%reordenar los elementos de funcion transformada. Calcular modulo (espectro de amplitud, "double sided) y fase
X=fftshift(X);  % [(0 + 0i), (0 + 0i), (0 + 0i) ,[-0 + 4i], (0 + 0i), [-0 - 4i], (0 + 0i), (0 + 0i)]             %ordenamos las frecuencias modo double sided. En el single sided hay que dividir por el numero de sample para que la amplitud sea correcta (1), ver siguiente ejemplo
P=(abs(X));     % [0,0,0,4,0,4,0,0]                                                                              %amplitud spectral, double sided. Es el modulo de cada número complejo. 
Phase=angle(X); % [0,0,0,(pi/2),0,(-1.570796326794897),0,0]                                                      %fase de cada número complejo;

figure; 

subplot(1,4,1);
stem(p); ylim([0 5]); xlim([0 10]); xlabel('k'); ylabel('Modulo'); 

subplot(1,4,2);
stem(f,P); ylim([0 5]); xlim([-5 5]); xlabel('Hz'); ylabel('Modulo'); 

subplot(1,4,3);
stem(f,Phase); ylim([-3 3]); xlim([-5 5]); xlabel('Hz'); ylabel('Fase');  

subplot(1,4,4);
plot(t,x,'-s'); xlabel('s'); ylabel('amplitud'); ylim([-1.2 1.2]); xlim([-0.05 1.05]); 


%% Q2: Display the single side amplitude frecuency spectrum for the  functions X1, X2, X3, X4
Fs=128;                    % Hz, sampling frequency
N=2^12;                    % number of samples
dt=1 / Fs;                 % s, resolucion en tiempo
df = Fs / N;               % Hz, resolución frecuencia
t=0:dt:(N-1)*dt;           % s, time points
f=-N/2*df:df:(N/2-1)*df;   % Hz,frequency points

x1=sin(2*pi*t);              % AU, señal X1
x2=x1; x2(floor(length(t)/2)+1:end)=0; %se establecen a cero las muestras a partir de la mitad del tiempo.
x3=x2; x3(floor(length(t)/4)+1:end)=0;
x4=x3; x4(floor(length(t)/8)+1:end)=0;

figure; 
subplot(4,1,1);
plot(t,x1);xlabel('s'); ylabel('amplitud');
subplot(4,1,2);
plot(t,x2);xlabel('s'); ylabel('amplitud');
subplot(4,1,3);
plot(t,x3);xlabel('s'); ylabel('amplitud');
subplot(4,1,4);
plot(t,x4);xlabel('s'); ylabel('amplitud');


X1=fft(x1); X2=fft(x2); X3=fft(x3); X4=fft(x4);
P1=abs(X1); P2=abs(X2); P3=abs(X3); P4=abs(X4);

P1 = fftshift(P1); P2 = fftshift(P2); P3 = fftshift(P3); P4 = fftshift(P4);  
P1(2:end)=2*P1(2:end)/N; P2(2:end)=2*P2(2:end)/N; P3(2:end)=2*P3(2:end)/N; P4(2:end)=2*P4(2:end)/N;

figure; 
subplot(4,1,1);
plot(f,P1); xlim([0 5]); xlabel('Hz'); ylabel('Amplitud');  
subplot(4,1,2);
plot(f,P2); xlim([0 5]); xlabel('Hz'); ylabel('Amplitud');  
subplot(4,1,3);
plot(f,P3); xlim([0 5]); xlabel('Hz'); ylabel('Amplitud'); 
subplot(4,1,4);
plot(f,P4); xlim([0 5]); xlabel('Hz'); ylabel('Amplitud'); 


%% Q3: Filter the central horizontal profile of the image shown below by setting to zero all the greatest 2/3 frecuencies. 
clc
close all
clear all 

load('reso.mat');
figure;
imagesc(reso);colormap('gray');

x=reso(size(reso,1)/2,:); % Se selecciona el perfil horizontal central

X=fft(x);            %calculating fft
X=fftshift(X);       %reordering frequencies

f=[-length(X)/2:1:(length(X)/2)-1];  %frequency values (Hz).

%filtering
FX=X;                                     % Crea una copia del vector X para aplicar el filtrado.            
FX(1:round(length(X)/3))=0;               % Pone a cero las frecuencias más bajas de la primera tercera parte del vector FX.
FX((length(X)-round(length(X)/3)):end)=0; % Pone a cero las frecuencias más altas de la última tercera parte del vector FX.

figure;
subplot(2,1,1);
plot(f,abs(X)); xlabel('Hz'); ylabel('Amplitude'); title('Not filtered');
subplot(2,1,2);
plot(f,abs(FX)); xlabel('Hz'); ylabel('Amplitude'); title('Filtered');

X=ifftshift(X);     %reordering back the frequencies 
x=real(ifft(X));    %inverse fourier (only real values are kept!!!!)


FX=ifftshift(FX);     %reordering back the frequencies 
fx=real(ifft(FX));    %inverse fourier (only real values are kept!!!!)

figure; 
subplot(2,1,1); 
plot(x);   xlabel('x (a.u)'); ylabel('Amplitude'); title('Not filtered'); ylim([-10 100]);
subplot(2,1,2);
plot(fx);   xlabel('x (a.u)'); ylabel('Amplitude'); title('filtered');    ylim([-10 100]);

reso_filtrada = reso;
[num_filas, num_columnas] = size(reso);

for i = 1:num_filas
    x = reso(i, :);
    X = fft(x);
    X = fftshift(X);

    FX = X;
    FX(1 : round(length(X) / 3)) = 0;
    FX((length(X) - round(length(X) / 3)) : end) = 0;

    fx = ifft(ifftshift(FX));
    reso_filtrada(i, :) = real(fx);
end

figure;
imagesc(reso_filtrada);
colormap('gray');
title('Imagen Filtrada (Detalle Fino Eliminado)');

%% Q4: Calculate the fourier transform over each row of the image "reso" and then over each column of the result. 
% Use the function fft2 over the image and compare
close all
clear all
clc

load('reso.mat');

X_2d_1=zeros(size(reso,1), size(reso,2));

[filas, columnas] = size(X_2d_1); 

for i = 1:filas
    X_2d_1(i,:) = fft(reso(i, :)); % Primero lo hace de reso
end

for j = 1:columnas
    X_2d_1(:,j) = fft(X_2d_1(:,j)); % Y despues de lo q queda
end

X = fft2(reso);

figure;
subplot(1,2,1);
imagesc(abs(fftshift(X_2d_1)),[0 100000]); xlabel('kx'); ylabel('ky');
subplot(1,2,2);
imagesc(abs(fftshift(X)),[0 100000]);  xlabel('kx'); ylabel('ky');


%% Q5 Display the 2 sided amplitude spectrum of a square function defined between 0 and 2 (seconds) 
% with a period of 1 Hz and a sampling frequency of 30 Hz. 
clc
close all
clear all

Fs=30;                   % Hz, sampling frequency
dt=1/Fs;                 % s, resolucion en tiempo
timelength=2*Fs*dt;      %time length of the signal
N=timelength/dt;         % number of samples
df=Fs/N;                 % Hz, resolución frecuencia
t=[0:dt:(N-1)*dt];       % s, time points
f=[-Fs/2:df:(Fs/2)-df];  % Hz,frequency points

x = square(t*2*pi);
figure; plot(t,x,'o'); xlabel('t'); ylabel('Amplitude');

X=fft(x);
X=fftshift(X);
P=(abs(X)); 
figure;
stem(f,P);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2. ESPACIO K
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Q1: Plot the a single a image corresponding to a k-space that has a single 
% real value corresponding to kx,ky. Obtain the corresponding images
clear all;
close all;
clc; 

kx=0;
ky=40; % sólo hay variación en el eje y
N = 251; 
M = 251;
space=zeros(N,M);

for(n=1:size(space,1))
    for(m=1:size(space,2))
    space(n,m)=(cos(2*pi*ky*n/size(space,1)))*(cos(2*pi*kx*m/size(space,2)))-(sin(2*pi*ky*n/size(space,1)))*(sin(2*pi*kx*m/size(space,2)));   
    end
end

imagesc(space); colormap('gray');

%% Q2: The file "Freso" contains the k-space of an actual MRI image. 
% Reconstruct the image by reconstructing each point of the k-space isolated 
% and sum up all the results to obtain the final image. Show the process 
% displaying each of the k-space points that is being reconstructed together 
% with the updated reconstructed image.  Hint: use a mask to isolate each point of the k-space.
%clear the workspace in each new execution

% realiza la reconstrucción de una imagen de resonancia magnética (MRI) a 
% partir de su espacio k, que es la representación de Fourier bidimensional de la imagen. 
% El proceso de reconstrucción se lleva a cabo punto por punto en el espacio 
% k y se visualiza la imagen reconstruida en tiempo real a medida que se actualiza.

clear all;
close all;

load('Freso.mat'); % contiene los datos del espacio k de la imagen MRI

imagen=zeros(size(Freso,1),size(Freso,2));        %create the variable that contains the updated reconstructed image.

ky=[-size(Freso,1)/2:1:(size(Freso,1)/2)-1];      %define ky values
kx=[-size(Freso,2)/2:1:(size(Freso,2)/2)-1];      %define kx values

mask=zeros(size(Freso,1),size(Freso,2));          %create the variable that contains the mask. All zeros.

figure; colormap('gray'); 

for(i=1:size(Freso,1))                            %create two loops in order to go through the whole k-space
    for(j=1:size(Freso,2))
        
        mask(:,:)=0.0;                            %refresh the mask
        mask(i,j)=1;                              %set to one the corresponding element of the k-space
        maskedF=Freso.*mask;                      %isolate the corresponding point of the k-space (masked k-space)
        invtrans=real(ifft2(ifftshift(maskedF))); %reconstruct such point.
        imagen=imagen+invtrans;                   %update the reconstructed image.
           
            if(mod(j,8)==0)                       %set a rule to display the image only for every 8 points the k-space in the x direction.
                subplot(1,2,1);                  
                imagesc(imagen); xlabel('x (a.u)'); ylabel('y (a.u)');         %display the updated reconstructed image.
                subplot(1,2,2);
                imagesc(kx,ky,abs(maskedF)); xlabel('kx (a.u)'); ylabel('ky (a.u)'); %display the point of the k-space that has been reconstructed.
                pause(0.001);
            end            
    end
end


% Este bucle anidado recorre cada punto del espacio k (i y j son los índices del espacio k). Para cada punto:
% 
% - Se reinicia la máscara a cero.
% - Se activa el punto (i,j) en la máscara.
% - Se aísla el punto (i,j) del espacio k multiplicando Freso por mask.
% - Se aplica la transformada inversa de Fourier al punto aislado para obtener la contribución en el espacio de imagen.
% - Se suma esta contribución a la imagen reconstruida.


%% Q3: Proceed like in Q1 but now show as well the reconstruction of each point of the k-space. 
% Display the images only for kx=0 and the rows of ky ranging from 40 to 88
clear all;
load('Freso.mat');

imagen=zeros(size(Freso,1),size(Freso,2));               %create the variable that contains the updated reconstructed image.

ky=[-size(Freso,1)/2:1:(size(Freso,1)/2)-1];             %define ky values
kx=[-size(Freso,2)/2:1:(size(Freso,2)/2)-1];             %define kx values

mask=zeros(size(Freso,1),size(Freso,2));                 %create the variable that contains the mask. All zeros.

figure; colormap("gray");

for(i=1:size(Freso,1))                                   %create two loops in order to go through the whole k-space.                                  
    for(j=1:size(Freso,2))
        
        mask(:,:)=0.0;                                   %refresh the mask
        mask(i,j)=1;                                     %set to one the corresponding element of the k-space.
        maskedF=Freso.*mask;                             %isolate the corresponding point of the k-space (masked k-space).
        invtrans=real(ifft2(ifftshift(maskedF)));        %reconstruct such point.
        imagen=imagen+invtrans;                          %update the reconstructed image.
            
            if(i>40 & i<88)                              %set a rule to display only the desired ky rows.
                if(j==size(Freso,2)/2+1)                 %set a rule to display only the images when kx=0.
                                                         % Esta condición
                                                         % selecciona el punto central de la matriz Freso en la dirección x, que corresponde a kx=0 en términos de frecuencias espaciales.

                    subplot(1,3,1);
                    imagesc(imagen);   xlabel('x (a.u)'); ylabel('y (a.u)');                   %display the updated image
                    subplot(1,3,2);
                    imagesc(invtrans); xlabel('x (a.u)'); ylabel('y (a.u)');                   %display the reconstructed point of the k-space. Labels?
                    subplot(1,3,3);
                    imagesc(kx,ky,abs(maskedF));xlabel('kx (a.u)'); ylabel('ky (a.u)');        %display the point of the k-space that has been reconstructed.
                    pause(1);
                end
            end
    end       
end

%% Recommendation for home: Do the same for diagonals of the k-space or ky=0.
clear all;
close all;
clc;

load('Freso.mat');
N = size(Freso, 1);
M = size(Freso, 2);
imagen = zeros(N, M);                                             % Variable para contener la imagen reconstruida.

kx = [-N/2:N/2];                                              % Define los valores de kx
ky = [-M/2:M/2];                                             % Define los valores de ky

mask = zeros(N, M);                                                  % Crea la máscara inicializada en ceros.

figure; colormap("gray");

for i = 1:N                                               % Itera sobre las filas de Freso                                  
    for j = 1:M                                           % Itera sobre las columnas de Freso
        
        mask = zeros(N, M);                                    % Reinicia la máscara
        mask(i, j) = 1;                                        % Establece a 1 el elemento correspondiente del espacio k
        maskedF = Freso .* mask;                               % Aisla el punto correspondiente del espacio k (espacio k enmascarado)
        invtrans = real(ifft2(ifftshift(maskedF)));            % Reconstruye ese punto
        imagen = imagen + invtrans;                            % Actualiza la imagen reconstruida
            
        if (i == floor(N/2)) && (j >= 40 && j <= 88)           % Establece una regla para mostrar solo las filas de ky deseadas
            subplot(1, 3, 1);
            imagesc(imagen);   xlabel('x (a.u)'); ylabel('y (a.u)');  % Muestra la imagen actualizada
            subplot(1, 3, 2);
            imagesc(invtrans);                                       % Muestra el punto reconstruido del espacio k
            subplot(1, 3, 3);
            imagesc(kx, ky, abs(maskedF));xlabel('kx (a.u)'); ylabel('ky (a.u)');  % Muestra el punto del espacio k que ha sido reconstruido
            pause(0.0001);
        end
    end       
end

%% Q4: Aliasing is a commont artifact in MRI. It arise from sub-sampling the k-space.
% Simulate a sub-sampled k-space from Freso by removing every even row of the k-space. 
% Reconstruct the corresponding image and observe the aspect of the artifact. Hint: use a counter if needed.
clear all;
clc;
close all;

load('Freso.mat');        
Freso_sub=zeros(size(Freso,1)/2,size(Freso,2));             %create the variable that will contain the sub-sampled k-space 
% matriz Freso_sub del mismo ancho que Freso, pero con la mitad del número de filas para almacenar el espacio k sub-muestreado.

count=1;

for(i=1:2:size(Freso,1))                                    %use a loop to populate the sub-sampled k-space from the correctly sampled k-space                         
    Freso_sub(count,:)=Freso(i,:);                           
    count=count+1;
end
% Se itera sobre las filas de Freso, tomando solo las filas impares (es decir, cada dos filas). 
% Estas filas se copian en Freso_sub, que se va llenando fila por fila. 
% Esto simula el sub-muestreo del espacio k eliminando las filas pares.



ky=[-size(Freso,1)/2:2:(size(Freso,1)/2)-1];              %define ky values
kx=[-size(Freso,2)/2:1:(size(Freso,2)/2)-1];              %define kx values
% El intervalo de ky se define con un paso de 2 para ajustarse al sub-muestreo.

figure; 
imagesc(kx,ky,abs(Freso_sub),[0 10000]); colormap('gray');   %display the sub-sampled k-space
xlabel('kx (a.u)'); ylabel('ky (a.u)');

imagen=real(ifft2(ifftshift(Freso_sub)));           %reconstruct the image

figure;
imagesc(imagen);colormap('gray');                  %display the reconstructed image
xlabel('x (a.u)'); ylabel('y (a.u)');

