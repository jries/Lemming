package org.lemming.outputs;

import org.lemming.data.Localization;
import org.lemming.input.SI;

public class PrintToScreen extends SI<Localization> {

	@Override
	public void process(Localization l) {
		System.out.println(String.format("%d, %f, %f",l.getID(),l.getX(),l.getY()));
	}
}
