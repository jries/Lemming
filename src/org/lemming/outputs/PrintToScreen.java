package org.lemming.outputs;

import org.lemming.data.Localization;
import org.lemming.interfaces.Well;

public class PrintToScreen implements Well<Localization> {

	@Override
	public void beforeRun() {
        }

	@Override
	public void afterRun() {
        }

	@Override
	public void process(Localization l) {
		System.out.println(String.format("%d, %f, %f",l.getID(),l.getX(),l.getY()));
	}
}
