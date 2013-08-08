package org.lemming.outputs;

import org.lemming.data.Localization;
import org.lemming.data.Store;
import org.lemming.interfaces.Output;

public class PrintToScreen implements Output, Runnable {

	Store<Localization> s;
	
	@Override
	public void setInput(Store<Localization> s) {
		this.s = s;
	}
	
	@Override
	public void run() {
		while(true) {
			Localization l = s.get();
			
			System.out.println(String.format("%d, %f, %f",l.getID(),l.getX(),l.getY()));
		}
	}
}
