package org.lemming.outputs;

import org.lemming.data.ImgLib2Frame;
import org.lemming.data.Localization;

import matlabcontrol.MatlabInvocationException;
import matlabcontrol.MatlabProxy;
import matlabcontrol.extensions.MatlabTypeConverter;

public class MatlabFeeder<T> extends SI<T> {

	MatlabProxy proxy;								// Proxy in charge of the communications with Matlab
    MatlabTypeConverter processor;					// Converter for variables between Matlab and Java
	
    int N; 											// Counter (of localizations or frames)
    
	public MatlabFeeder(MatlabProxy p){
		N = 1; 											// In matlab counts begin at 1
		proxy = p;
		processor = new MatlabTypeConverter(proxy);
	}
	
	@Override
	public void process(T element) {
		
		// Test what the type of the element is
		if(element instanceof Localization){
			// Add X and Y to a matlab vector
			try {
				proxy.setVariable("xval", ((Localization) element).getX());			// Matlab buffer variable for x positions
				proxy.setVariable("yval", ((Localization) element).getY());
				proxy.setVariable("n",N);											// Index of the localization
				proxy.eval("X(n,1)=xval;");											// Add to vector
				proxy.eval("Y(n,1)=yval;");
			} catch (MatlabInvocationException e1) {
				e1.printStackTrace();
			}
			
			N++;
			
		} else if(element instanceof ImgLib2Frame){
			
			
		}
		
	}

}
