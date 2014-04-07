package org.lemming.inputs;

import javax.swing.JOptionPane;

import matlabcontrol.MatlabConnectionException;
import matlabcontrol.MatlabInvocationException;
import matlabcontrol.MatlabProxy;
import matlabcontrol.MatlabProxyFactory;
import matlabcontrol.MatlabProxyFactoryOptions;
import matlabcontrol.extensions.MatlabTypeConverter;

import org.lemming.inputs.SO;

public class MatlabSOController<T> extends SO<T>  {

	MatlabProxyFactoryOptions options;				// Proxy constructor options 
	MatlabProxyFactory factory;						// Proxy factory
	MatlabProxy proxy;								// Proxy in charge of the communications with Matlab
    MatlabTypeConverter processor;					// Converter for variables between Matlab and Java
	
	private MatlabGrabber grabber;

	String path;
	String matcommand;
	Boolean launcher;
	Boolean user_done = false;
	String type;
	
	public MatlabSOController(String t) throws MatlabConnectionException, MatlabInvocationException{
		launcher = false;
		type = t;
		options = new MatlabProxyFactoryOptions.Builder()
        .setUsePreviouslyControlledSession(true)
        .build();	
		
		factory = new MatlabProxyFactory(options);
		proxy = factory.getProxy();												// Create proxy between JVM and Matlab
	    processor = new MatlabTypeConverter(proxy);								

	    grabber = new MatlabGrabber<T>(proxy,type);
	}	
	
	public MatlabSOController(String s1, String s2, String t) throws MatlabConnectionException, MatlabInvocationException{
		launcher = true;
		path = s1;
		matcommand = s2;
		type = t;
		options = new MatlabProxyFactoryOptions.Builder()
        .setUsePreviouslyControlledSession(true)
        .build();	
		
		factory = new MatlabProxyFactory(options);
		proxy = factory.getProxy();												// Create proxy between JVM and Matlab
	    processor = new MatlabTypeConverter(proxy);								

	    grabber = new MatlabGrabber<T>(proxy,type);
	}
		
	public void beforeRun() {
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
		grabber.beforeRun();
	}
	
	public void afterRun(){
		disconnectProxy();
	}
	
	@Override
	public boolean hasMoreOutputs() {
		return grabber.hasMoreOutputs();
	}

	@Override
	protected T newOutput() {
		return (T) grabber.newOutput();
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
}
