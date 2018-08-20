#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
using namespace std;

int main() {
   srand(time(NULL));
   char cadena[128];
   string data;
   string mensaje="topology: 4 5 3 3\n",x,y,w,z,a,b,c,d,m,n,q,p;
   //_________________Iris Setosa________________
   int Alimite_inferiorsp=47,Alimite_superiorsp=53;
   int Alimite_inferiorsw=36,Alimite_superiorsw=38;
   int Alimite_inferiorpl=15,Alimite_superiorpl=18;
   int Alimite_inferiorpw=2,Alimite_superiorpw=5;
  // double sepallength,sepalwidth,petallength,petalwidth;
   //_____________________________________________
   //_________________Iris Versicolor________________
   int Blimite_inferiorsp=58,Blimite_superiorsp=63;
   int Blimite_inferiorsw=25,Blimite_superiorsw=28;
   int Blimite_inferiorpl=40,Blimite_superiorpl=46;
   int Blimite_inferiorpw=11,Blimite_superiorpw=15;
   //double sepallength,sepalwidth,petallength,petalwidth;
   //_____________________________________________
   //_________________Iris Virginica________________
   int Climite_inferiorsp=60,Climite_superiorsp=70;
   int Climite_inferiorsw=28,Climite_superiorsw=32;
   int Climite_inferiorpl=48,Climite_superiorpl=64;
   int Climite_inferiorpw=16,Climite_superiorpw=24;
   double sepallength,sepalwidth,petallength,petalwidth;
   //_____________________________________________
   // Crea un fichero de salida
   ofstream fs("trainingData4.txt");
   ofstream qfs("data.txt");  
   
   // Enviamos una cadena al fichero de salida:
   for(int i=1;i<=30000;i++){
   	//#########################SETOSA#########################################################
   	if(i>=1 && i<=10000){
   	//sepal lengh caclulo de valor aleatorio------------------------
   	sepallength = (float)(Alimite_inferiorsp + rand() % (Alimite_superiorsp +1 - Alimite_inferiorsp)) ;
   	sepallength=sepallength/10;
   	x=to_string(sepallength); //resultado
   	x=x.substr(0,3); //resultado
   	//-------------------------------------------
   	//---------------Sepal Lengh caculo de valor aleatorio
   	sepalwidth = (float)(Alimite_inferiorsw + rand() % (Alimite_superiorsw +1 - Alimite_inferiorsw)) ;
   	sepalwidth=sepalwidth/10;
   	y=to_string(sepalwidth); //r esultado
   	y=y.substr(0,3); //resultado
   	//-------------------------------------------------------
   	//---------------Petal Lengh caculo de valor aleatorio
   	petallength = (float)(Alimite_inferiorpl + rand() % (Alimite_superiorpl +1 - Alimite_inferiorpl)) ;
   	petallength=petallength/10;
   	w=to_string(petallength); //r esultado
   	w=w.substr(0,3); //resultado
   	//-------------------------------------------------------
   	//---------------Petal Width caculo de valor aleatorio
   	petalwidth = (float)(Alimite_inferiorpw + rand() % (Alimite_superiorpw +1 - Alimite_inferiorpw)) ;
   	petalwidth=petalwidth/10;
   	z=to_string(petalwidth); //resultado
   	z=z.substr(0,3); //resultado
   	mensaje=mensaje+"in: "+x+" "+y+" "+w+" "+z+" \n"+"out: 1.0 0.0 0.0\n";
   	data=data+x+","+y+","+w+","+z+","+"Iris-setosa\n";
   }
   	//-------------------------------------------------------
   	//#####################################################################################
   	//##############################VERSICOLOR###########################################
   	//sepal lengh caclulo de valor aleatorio------------------------
   if(i>=10001 && i<=20000){
   	sepallength = (float)(Blimite_inferiorsp + rand() % (Blimite_superiorsp +1 - Blimite_inferiorsp)) ;
   	sepallength=sepallength/10;
   	a=to_string(sepallength); //resultado
   	a=a.substr(0,3); //resultado
   	//-------------------------------------------
   	//---------------Sepal Lengh caculo de valor aleatorio
   	sepalwidth = (float)(Blimite_inferiorsw + rand() % (Blimite_superiorsw +1 - Blimite_inferiorsw)) ;
   	sepalwidth=sepalwidth/10;
   	b=to_string(sepalwidth); //r esultado
   	b=b.substr(0,3); //resultado
   	//-------------------------------------------------------
   	//---------------Petal Lengh caculo de valor aleatorio
   	petallength = (float)(Blimite_inferiorpl + rand() % (Blimite_superiorpl +1 - Blimite_inferiorpl)) ;
   	petallength=petallength/10;
   	c=to_string(petallength); //r esultado
   	c=c.substr(0,3); //resultado
   	//-------------------------------------------------------
   	//---------------Petal Width caculo de valor aleatorio
   	petalwidth = (float)(Blimite_inferiorpw + rand() % (Blimite_superiorpw +1 - Blimite_inferiorpw)) ;
   	petalwidth=petalwidth/10;
   	d=to_string(petalwidth); //resultado
   	d=d.substr(0,3);
   	mensaje=mensaje+"in: "+a+" "+b+" "+c+" "+d+" \n"+"out: 0.0 1.0 0.0\n";
   	data=data+a+","+b+","+c+","+d+","+"Iris-versicolor\n"; 
   	} //resultado
   	//-------------------------------------------------------
   	//#####################################################################################
   	//##############################VIRGINICA###########################################
   	if(i>=20001 && i<=30000){
   	//sepal lengh caclulo de valor aleatorio------------------------
   	sepallength = (float)(Climite_inferiorsp + rand() % (Climite_superiorsp +1 - Climite_inferiorsp)) ;
   	sepallength=sepallength/10;
   	n=to_string(sepallength); //resultado
   	n=n.substr(0,3); //resultado
   	//-------------------------------------------
   	//---------------Sepal Lengh caculo de valor aleatorio
   	sepalwidth = (float)(Climite_inferiorsw + rand() % (Climite_superiorsw +1 - Climite_inferiorsw)) ;
   	sepalwidth=sepalwidth/10;
   	m=to_string(sepalwidth); //r esultado
   	m=m.substr(0,3); //resultado
   	//-------------------------------------------------------
   	//---------------Petal Lengh caculo de valor aleatorio
   	petallength = (float)(Climite_inferiorpl + rand() % (Climite_superiorpl +1 - Climite_inferiorpl)) ;
   	petallength=petallength/10;
   	p=to_string(petallength); //r esultado
   	p=p.substr(0,3); //resultado
   	//-------------------------------------------------------
   	//---------------Petal Width caculo de valor aleatorio
   	petalwidth = (float)(Climite_inferiorpw + rand() % (Climite_superiorpw +1 - Climite_inferiorpw)) ;
   	petalwidth=petalwidth/10;
   	q=to_string(petalwidth); //resultado
   	q=q.substr(0,3); //resultado
   	mensaje=mensaje+"in: "+m+" "+n+" "+p+" "+q+" \n"+"out: 0.0 0.0 1.0\n";
   	data=data+n+","+m+","+p+","+q+","+"Iris-virginica\n";  
   }
   	//#####################################################################################

   	 
   	
   	   
   }

   fs << mensaje << endl;
   qfs << data << endl;
   // Cerrar el fichero, 
   // para luego poder abrirlo para lectura:
   fs.close();
   qfs.close();

   // Abre un fichero de entrada
   //ifstream fe("nombre.txt"); 

   // Leeremos mediante getline, si lo hiciéramos 
   // mediante el operador << sólo leeríamos 
   // parte de la cadena:
   //fe.getline(cadena, 128);

  // cout << cadena << endl;

   return 0;
}