package org.lemming.matlab;

import javax.swing.JOptionPane;

import org.lemming.outputs.SI;

import matlabcontrol.MatlabConnectionException;
import matlabcontrol.MatlabInvocationException;
import matlabcontrol.MatlabProxy;
import matlabcontrol.MatlabProxyFactory;
import matlabcontrol.MatlabProxyFactoryOptions;
import matlabcontrol.extensions.MatlabTypeConverter;

public class MatlabSIController<T> extends SI<T> {

	MatlabProxyFactoryOptions options;				// Proxy constructor options 
	MatlabProxyFactory factory;						// Proxy factory
	MatlabProxy proxy;								// Proxy in charge of the communications with Matlab
    MatlabTypeConverter processor;					// Converter for variables between Matlab and Java
	
	private MatlabFeeder feeder;

	String path;
	String matcommand;
	Boolean launcher;
	Boolean user_done = false;
	
	public MatlabSIController() throws MatlabConnectionException, MatlabInvocationException{
		launcher = false;
		options = new MatlabProxyFactoryOptions.Builder()
        .setUsePreviouslyControlledSession(true)
        .build();	
		
		factory = new MatlabProxyFactory(options);
		proxy = factory.getProxy();												// Create proxy between JVM and Matlab
	    processor = new MatlabTypeConverter(proxy);								

	    feeder = new MatlabFeeder<T>(proxy);
	}
	
	public MatlabSIController(String s1, String s2) throws MatlabConnectionException, MatlabInvocationException{
		launcher = true;
		path = s1;
		matcommand = s2;
		options = new MatlabProxyFactoryOptions.Builder()
        .setUsePreviouslyControlledSession(true)
        .build();	
		
		factory = new MatlabProxyFactory(options);
		proxy = factory.getProxy();												// Create proxy between JVM and Matlab
	    processor = new MatlabTypeConverter(proxy);								

	    feeder = new MatlabFeeder<T>(proxy);
	}
	
	public void afterRun(){
		if(launcher){
			try {
				runMatfile();
			} catch (MatlabInvocationException e) {
				e.printStackTrace();
			}
			
			// Wait until user is done
			JOptionPane.showMessageDialog ( 									// Dialog to pause the program 
					null, "Waiting for end of MATLAB task" );

		}
		user_done = true;
	}
	
	@Override
	public void process(T element) {
		
		feeder.process(element);
	
		if(input.isEmpty()){
						
			disconnectProxy();
		}	
	}

	public void disconnectProxy(){
		proxy.disconnect();
	}

	void runMatfile() throws MatlabInvocationException{
		proxy.eval("addpath "+path);
		proxy.eval(matcommand);
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
