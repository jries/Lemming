package org.lemming.outputs;

import java.util.Arrays;

import org.lemming.data.Localization;
import org.lemming.data.Store;
import org.lemming.interfaces.Output;

public class PrintToScreen implements Output {

	Store<Localization> s;

	public PrintToScreen() {};

	public PrintToScreen(Object[] a) {
		System.out.println(Arrays.deepToString(a));
	}

	public PrintToScreen(double[] a) {
		System.out.println(Arrays.toString(a));
	}

	@Override
	public void run() {		
		if (s==null) {new NullStoreWarning(this.getClass().getName()); return;}		
		while(true) {
			Localization l = s.get();			
			System.out.println(String.format("%d, %f, %f",l.getID(),l.getX(),l.getY()));
		}		
	}

	@Override
	public void setInput(Store<Localization> s) {
		this.s = s;
	}
	
}
