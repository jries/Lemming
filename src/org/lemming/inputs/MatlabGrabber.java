package org.lemming.inputs;

import matlabcontrol.MatlabInvocationException;
import matlabcontrol.MatlabProxy;
import matlabcontrol.extensions.MatlabTypeConverter;

import org.lemming.data.XYLocalization;

public class MatlabGrabber<T> extends SO<T> {

	MatlabProxy proxy;								// Proxy in charge of the communications with Matlab
    MatlabTypeConverter processor;					// Converter for variables between Matlab and Java
	
    int N; 											// Counter (of localizations or frames)   
    int counter;
    
    double[][] xloc;
    double[][] yloc;
    
    String type;
    
	public MatlabGrabber(MatlabProxy p, String t){
		proxy = p;
		processor = new MatlabTypeConverter(proxy);
		type = t;
		counter = 0;
	}

	public void beforeRun(){
		if(type == "Localization"){
			// Get localizations back from matlab
		    try {
		    	double length = ((double[]) proxy.getVariable("length(X)"))[0];				//!!! can be done through matlabarray.getlength() or in java once X is retrieved
		    	N = (int) length;
		    	//System.out.println(final_length);
				xloc = processor.getNumericArray("X").getRealArray2D();	// Matlab arrays are always 2D
				yloc = processor.getNumericArray("Y").getRealArray2D();
			} catch (MatlabInvocationException e) {
				e.printStackTrace();
			}
			
		}else if(type == "ImgLib2Frame"){
			
			
		}
		
	}

	@Override
	public boolean hasMoreOutputs() {
		if(counter == N-1){
			return false;
		}
		return true;
	}

	@Override
	public T newOutput() {																/// here newOutput has been change to public
		if(type == "Localization"){
			int i = counter;
			counter++;
			return (T) new XYLocalization(xloc[i][0],yloc[i][0]);
		} else if(type == "ImgLib2Frame"){
			
		}
		return null;
	}

}
