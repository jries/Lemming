package org.lemming.matlab;

import javax.swing.JOptionPane;

import org.lemming.processors.SISO;

import matlabcontrol.MatlabConnectionException;
import matlabcontrol.MatlabInvocationException;
import matlabcontrol.MatlabProxy;
import matlabcontrol.MatlabProxyFactory;
import matlabcontrol.MatlabProxyFactoryOptions;
import matlabcontrol.extensions.MatlabTypeConverter;

public class MatlabSISOController<T1,T2> extends SISO<T1,T2> {

	String path;									// Local path to the directory of the matlab program
	String matcommand;							// Name of the matlab file .m (should be a command)

	MatlabProxyFactoryOptions options;				// Proxy constructor options 
	MatlabProxyFactory factory;						// Proxy factory
	MatlabProxy proxy;								// Proxy in charge of the communications with Matlab
    MatlabTypeConverter processor;					// Converter for variables between Matlab and Java
    
	int N=1;										// Counter for the localizations
	
	Boolean user_done = false;
	Boolean launcher;
	
	String type1;
	//String type2;
	
	MatlabFeeder<T1> feeder;
	MatlabGrabber<T2> grabber;
	
	public MatlabSISOController(String t) throws MatlabConnectionException{
		launcher = false;
		type1 = t;
		options = new MatlabProxyFactoryOptions.Builder()
        .setUsePreviouslyControlledSession(true)
        .build();	
		
		factory = new MatlabProxyFactory(options);
		proxy = factory.getProxy();												// Create proxy between JVM and Matlab
	    processor = new MatlabTypeConverter(proxy);

	    feeder = new MatlabFeeder<T1>(proxy);
	    grabber = new MatlabGrabber<T2>(proxy,type1);
	}
	
	public MatlabSISOController(String t, String s1, String s2) throws MatlabConnectionException{
		launcher = true;
		type1 = t;
		path = s1;
		matcommand = s2;
		options = new MatlabProxyFactoryOptions.Builder()
        .setUsePreviouslyControlledSession(true)
        .build();	
		
		factory = new MatlabProxyFactory(options);
		proxy = factory.getProxy();												// Create proxy between JVM and Matlab
	    processor = new MatlabTypeConverter(proxy);

	    feeder = new MatlabFeeder<T1>(proxy);
	    grabber = new MatlabGrabber<T2>(proxy,type1);
	}
	
	@Override
	public void process(T1 element) {
		feeder.process(element);
		
		if(input.isEmpty()){
			try {
				runMatfile();
			} catch (MatlabInvocationException e) {
				e.printStackTrace();
			}
			
			grabber.beforeRun();
			
			while(grabber.hasMoreOutputs()){
				output.put(grabber.newOutput());
			}
			
			this.disconnectProxy();
		}
		
	}
	
	public void disconnectProxy(){
		proxy.disconnect();
	}
	
	public void runMatfile() throws MatlabInvocationException{
		proxy.eval("addpath "+path);
		proxy.eval(matcommand);
		
		// Wait until user is done
		JOptionPane.showMessageDialog ( 									// Dialog to pause the program 
				null, "Waiting for end of MATLAB task" );
		
		user_done = true;
	}
	
	public boolean isUserDone() {
		return user_done;
	}

	public void execute(String s) throws MatlabInvocationException{
		if(proxy.isConnected()){
			proxy.eval(s);
		}
	}
}
