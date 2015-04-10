package org.lemming.outputs;

import org.lemming.interfaces.Localization;

/**
 * @author Ronny Sczech
 *
 */
public class PrintToScreen extends SingleInput<Localization> {

	@Override
	public void process(Localization l) {
		if (l==null) return;
		System.out.println(String.format("%d, %f, %f",l.getID(),l.getX(),l.getY()));
		if (l.isLast())
			stop();
	}

}
